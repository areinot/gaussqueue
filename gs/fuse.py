"""Multi-view LiDAR depth fusion: merge every photo's depth map into one 3D model, trusting
each reading more the closer and more confident it was, then render the model back into
every camera. Photos get denoised depth, plus depth for surfaces they saw from beyond the
LiDAR's range that were measured up close from somewhere else in the capture.

  python fuse.py <data> --depth-dir <sensor_data> --out <dir> [--exclude-every N]
                 [--voxel 0.02] [--trunc 0.06] [--max-range 5]

<data> holds images/ and a COLMAP text model (refined poses). Writes, per photo, the same
files SplatKing's sensor_data has (<stem>_depth.bin float32 metres, <stem>_confidence.bin
uint8, <stem>.json), so train.py takes <out> as its --depth-dir: confidence 2 where the
fused model (or, failing that, the photo's own confident LiDAR) gives depth, else 0.
Also writes fusion.json with coverage and agreement statistics.

Method: a TSDF volume (truncated signed distance, as in KinectFusion) on the GPU. Each
reading is weighted by ARKit's confidence and by 1/depth^2 (iPhone LiDAR noise grows with
range), so close views dominate. Free space in front of each reading is carved too, which
removes floaters. Rendering ray-marches each photo's depth pixels to the first surface, so
a far camera only gets depth for what it can actually see.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import gsdata

CONF_WEIGHT = {0: 0.0, 1: 0.3, 2: 1.0}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data")
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--voxel", type=float, default=0.02, help="voxel size in metres (grown if the scene is large)")
    p.add_argument("--trunc", type=float, default=0.06, help="truncation distance in metres")
    p.add_argument("--max-range", type=float, default=5.0, help="ignore LiDAR readings beyond this")
    p.add_argument("--max-voxels", type=float, default=2.5e8)
    p.add_argument("--min-weight", type=float, default=0.15,
                   help="fused depth needs at least this much evidence (1 = one confident reading at 1 m)")
    p.add_argument("--agree", type=float, default=0.10,
                   help="relative tolerance between fused depth and the photo's own reading")
    p.add_argument("--grid-factor", type=int, default=4,
                   help="space grid cell = this many fusion voxels (default 4; 1 for small, detailed boxes)")
    p.add_argument("--grid-only", action="store_true",
                   help="only write space.npz (known-empty / surface / unknown), no depth maps")
    p.add_argument("--exclude-every", type=int, default=0,
                   help="leave every Nth photo out of the fusion (the trainer's held-out photos)")
    return p.parse_args(argv)


def load(root, depth_dir, images):
    """Per photo: name, world-to-camera [4,4], intrinsics at depth resolution, depth, confidence."""
    frames = []
    for im in images:
        d, c = gsdata.read_depth(depth_dir, im["name"])
        if d is None:
            continue
        cam, (h, w) = im["cam"], d.shape
        sx, sy = w / cam["w"], h / cam["h"]
        frames.append({"name": im["name"], "vm": torch.tensor(im["viewmat"], dtype=torch.float32),
                       "K": (cam["fx"] * sx, cam["fy"] * sy, cam["cx"] * sx, cam["cy"] * sy),
                       "depth": d, "conf": c})
    return frames


def backproject(f, mask):
    fx, fy, cx, cy = f["K"]
    v, u = torch.nonzero(mask, as_tuple=True)
    z = f["depth"][v, u]
    xc = torch.stack([(u + 0.5 - cx) / fx * z, (v + 0.5 - cy) / fy * z, z], 1)
    inv = torch.linalg.inv(f["vm"])
    return xc @ inv[:3, :3].T + inv[:3, 3]


def bounds(frames, max_range):
    pts = []
    for f in frames:
        m = (f["conf"] >= 1) & (f["depth"] > 0.05) & (f["depth"] <= max_range)
        p = backproject(f, m)
        pts.append(p[torch.randperm(len(p))[:20000]])
    pts = torch.cat(pts)
    lo, hi = torch.quantile(pts, 0.002, dim=0), torch.quantile(pts, 0.998, dim=0)
    return lo - 0.25, hi + 0.25


class Volume:
    def __init__(self, lo, hi, voxel, trunc, max_voxels, dev):
        ext = (hi - lo)
        voxel = max(voxel, float((ext.prod() / max_voxels) ** (1 / 3)))
        self.voxel, self.trunc = voxel, max(trunc, 3 * voxel)
        self.lo = lo.to(dev)
        self.n = [int(math.ceil(float(e) / voxel)) for e in ext]   # x, y, z counts
        nx, ny, nz = self.n
        self.tsdf = torch.ones(nz, ny, nx, device=dev)
        self.w = torch.zeros(nz, ny, nx, device=dev)
        self.dev = dev

    def integrate(self, f, max_range, slab=16):
        dev = self.dev
        vm = f["vm"].to(dev)
        fx, fy, cx, cy = f["K"]
        depth, conf = f["depth"].to(dev), f["conf"].to(dev)
        h, w = depth.shape
        cw = torch.tensor([CONF_WEIGHT[0], CONF_WEIGHT[1], CONF_WEIGHT[2]], device=dev)
        good = (depth > 0.05) & (depth <= max_range) & (conf >= 1)
        nx, ny, nz = self.n
        xs = self.lo[0] + (torch.arange(nx, device=dev) + 0.5) * self.voxel
        ys = self.lo[1] + (torch.arange(ny, device=dev) + 0.5) * self.voxel
        for k0 in range(0, nz, slab):
            k1 = min(nz, k0 + slab)
            zs = self.lo[2] + (torch.arange(k0, k1, device=dev) + 0.5) * self.voxel
            Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")
            P = torch.stack([X, Y, Z], -1)
            Pc = P @ vm[:3, :3].T + vm[:3, 3]
            z = Pc[..., 2]
            u = torch.floor(Pc[..., 0] / z.clamp_min(1e-6) * fx + cx).long()
            v = torch.floor(Pc[..., 1] / z.clamp_min(1e-6) * fy + cy).long()
            inside = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not inside.any():
                continue
            uu, vv = u.clamp(0, w - 1), v.clamp(0, h - 1)
            d = depth[vv, uu]
            ok = inside & good[vv, uu]
            sdf = d - z
            ok &= sdf > -self.trunc
            if not ok.any():
                continue
            s = (sdf / self.trunc).clamp(max=1.0)
            wt = cw[conf[vv, uu].long()] / d.clamp_min(0.3) ** 2
            wt = torch.where(ok, wt, torch.zeros_like(wt))
            T, W = self.tsdf[k0:k1], self.w[k0:k1]
            Wn = W + wt
            self.tsdf[k0:k1] = torch.where(ok, (T * W + s * wt) / Wn.clamp_min(1e-12), T)
            self.w[k0:k1] = Wn

    def space_grid(self, factor=4, min_weight=0.2, min_share=0.75):
        """Coarse map of what the LiDAR established: 1 = known empty (most of the block seen
        through repeatedly, none of it surface), 2 = surface (or next to one), 0 = unknown.
        Foliage, where some rays hit and some pass, averages below "empty"."""
        T, W = self.tsdf, self.w
        nz, ny, nx = T.shape
        pad = [0, (-nx) % factor, 0, (-ny) % factor, 0, (-nz) % factor]
        empty = F.pad(((W >= min_weight) & (T >= 0.95)).float(), pad)
        surf = F.pad(((W > 0) & (T.abs() < 0.9)).float(), pad)
        empty_c = (F.avg_pool3d(empty[None, None], factor)[0, 0] >= min_share).float()
        surf_c = F.max_pool3d(surf[None, None], factor)[0, 0]
        surf_c = F.max_pool3d(surf_c[None, None], 3, stride=1, padding=1)[0, 0]   # 1-voxel margin
        grid = torch.zeros_like(surf_c, dtype=torch.uint8)
        grid[empty_c > 0] = 1
        grid[surf_c > 0] = 2
        return {"grid": grid.cpu().numpy(), "origin": self.lo.cpu().numpy(),
                "voxel": np.float32(self.voxel * factor)}

    def sample(self, P):
        """Trilinear TSDF and weight at world points P [N,3]."""
        nx, ny, nz = self.n
        idx = (P - self.lo) / self.voxel - 0.5
        g = torch.stack([idx[:, 0] / max(nx - 1, 1), idx[:, 1] / max(ny - 1, 1), idx[:, 2] / max(nz - 1, 1)], 1) * 2 - 1
        g = g.view(1, -1, 1, 1, 3)
        if getattr(self, "packed", None) is None:   # built once, after integration
            self.packed = torch.stack([self.tsdf, self.w])[None]
        out = F.grid_sample(self.packed, g, mode="bilinear", padding_mode="border", align_corners=True)
        out = out.view(2, -1)
        inb = (g.view(-1, 3).abs() <= 1).all(1)
        return out[0], torch.where(inb, out[1], torch.zeros_like(out[1]))

    def render(self, f, min_weight, far):
        """Fused z-depth and evidence per depth pixel (0 where no surface was found)."""
        dev = self.dev
        fx, fy, cx, cy = f["K"]
        h, w = f["depth"].shape
        vv, uu = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
        dirs = torch.stack([(uu + 0.5 - cx) / fx, (vv + 0.5 - cy) / fy, torch.ones_like(uu, dtype=torch.float32)], -1).view(-1, 3)
        inv = torch.linalg.inv(f["vm"].to(dev))
        dw = dirs @ inv[:3, :3].T
        o = inv[:3, 3]
        step = self.trunc * 0.5
        n = dirs.shape[0]
        depth = torch.zeros(n, device=dev)
        wout = torch.zeros(n, device=dev)
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        prev_s = torch.ones(n, device=dev)
        prev_w = torch.zeros(n, device=dev)
        zprev = 0.1
        zt = 0.1 + step
        while zt < far and not bool(done.all()):
            s, wt = self.sample(o + dw * zt)
            cross = (~done) & (prev_s > 0) & (s <= 0) & (prev_w >= min_weight) & (wt >= min_weight)
            if cross.any():
                frac = prev_s / (prev_s - s).clamp_min(1e-6)
                depth = torch.where(cross, zprev + frac * (zt - zprev), depth)
                wout = torch.where(cross, torch.minimum(prev_w, wt), wout)
            # a confident negative region without a crossing means we started inside: give up
            done |= cross | ((s < 0) & (wt >= min_weight))
            prev_s, prev_w, zprev = s, wt, zt
            zt += step
        return depth.view(h, w), wout.view(h, w)


def main(a):
    gsdata.keep_awake()
    t0 = time.time()
    dev = "cuda"
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images, _, _ = gsdata.read_model(gsdata.find_model(root))
    _, held = gsdata.split(images, a.exclude_every)
    held_names = {im["name"] for im in held}
    frames = load(root, a.depth_dir, images)
    fuse_frames = [f for f in frames if f["name"] not in held_names]
    lo, hi = bounds(fuse_frames, a.max_range)
    vol = Volume(lo, hi, a.voxel, a.trunc, a.max_voxels, dev)
    print(f"{len(fuse_frames)} depth maps fused ({len(held_names)} held out), volume "
          f"{vol.n[0]}x{vol.n[1]}x{vol.n[2]} at {100 * vol.voxel:.1f} cm, extent "
          f"{(hi - lo).numpy().round(1).tolist()} m", flush=True)
    for f in fuse_frames:
        vol.integrate(f, a.max_range)
    t1 = time.time()
    print(f"integrated in {t1 - t0:.0f}s", flush=True)
    space = vol.space_grid(factor=a.grid_factor)
    np.savez_compressed(out / "space.npz", **space)
    g = space["grid"]
    print(f"space grid {g.shape[2]}x{g.shape[1]}x{g.shape[0]} at {100 * float(space['voxel']):.1f} cm: "
          f"{100 * (g == 1).mean():.0f}% known empty, {100 * (g == 2).mean():.0f}% surface", flush=True)
    if a.grid_only:
        return {"space": {"empty": float((g == 1).mean()), "surface": float((g == 2).mean())}}

    far = float((hi - lo).norm()) + 1
    stats = {"frames": [], "voxel_m": vol.voxel, "trunc_m": vol.trunc, "fused_frames": len(fuse_frames)}
    for f in frames:
        fd, fw = vol.render(f, a.min_weight, far)
        fd, fw = fd.cpu(), fw.cpu()
        raw, rc = f["depth"], f["conf"]
        raw_ok = (rc == 2) & (raw > 0.05) & (raw <= a.max_range)
        fused_ok = fd > 0
        # Consistency gate. The fused model only holds surfaces some photo measured up close;
        # foliage or a branch nobody got near is missing, and the fused surface behind it would
        # push it away. So: where this photo's own confident LiDAR agrees, use the (denoised)
        # fused depth; where it disagrees, trust the photo. Fill pixels the photo has no
        # confident reading for only when its own low-confidence reading doesn't see something
        # clearly closer than the fused surface.
        agree = (fd - raw).abs() <= a.agree * raw
        denoise = raw_ok & fused_ok & agree
        own = raw_ok & ~denoise
        fill = ~raw_ok & fused_ok & (raw > 0.05) & (fd <= raw * (1 + a.agree))
        use_fused = denoise | fill
        sup = torch.where(use_fused, fd, torch.where(own, raw, torch.zeros_like(raw)))
        conf = ((use_fused | own).to(torch.uint8) * 2)
        occluded = ~raw_ok & fused_ok & (raw > 0.05) & (fd > raw * (1 + a.agree))
        stem = Path(f["name"]).stem
        sup.numpy().astype("<f4").tofile(out / f"{stem}_depth.bin")
        conf.numpy().astype(np.uint8).tofile(out / f"{stem}_confidence.bin")
        meta = Path(a.depth_dir) / f"{stem}.json"
        if meta.exists():
            (out / f"{stem}.json").write_text(meta.read_text())
        # agreement with the photo's own most reliable LiDAR (confident and within 3 m)
        near = raw_ok & (raw <= 3.0) & fused_ok
        err = ((fd[near] - raw[near]).abs() / raw[near]) if near.any() else torch.zeros(0)
        stats["frames"].append({
            "name": f["name"], "held_out": f["name"] in held_names,
            "raw_coverage": float(raw_ok.float().mean()), "fused_coverage": float(fused_ok.float().mean()),
            "supervised": float((conf > 0).float().mean()),
            "beyond_raw": float(fill.float().mean()),
            "rejected_occluded": float(occluded.float().mean()),
            "kept_own": float((own & fused_ok).float().mean()),
            "near_agreement_median": float(err.median()) if len(err) else None,
            "near_agreement_p90": float(torch.quantile(err, 0.9)) if len(err) > 10 else None})
    fr = stats["frames"]

    def mean(key, sel):
        v = [x[key] for x in fr if sel(x) and x[key] is not None]
        return sum(v) / len(v) if v else None
    for label, sel in (("all", lambda x: True), ("held_out", lambda x: x["held_out"])):
        stats[label] = {k: mean(k, sel) for k in ("raw_coverage", "fused_coverage", "supervised", "beyond_raw",
                                                   "rejected_occluded", "kept_own",
                                                   "near_agreement_median", "near_agreement_p90")}
    stats["seconds"] = round(time.time() - t0, 1)
    (out / "fusion.json").write_text(json.dumps(stats, indent=1))
    s = stats["all"]
    print(f"rendered in {time.time() - t1:.0f}s. Pixels with depth: raw confident {100 * s['raw_coverage']:.0f}% -> "
          f"supervised {100 * s['supervised']:.0f}% (fused {100 * s['fused_coverage']:.0f}%, "
          f"{100 * s['beyond_raw']:.0f}% new, {100 * s['rejected_occluded']:.0f}% rejected as occluded)", flush=True)
    if stats["held_out"]["near_agreement_median"] is not None:
        h = stats["held_out"]
        print(f"held-out photos (not fused): fused vs their own confident LiDAR within 3 m: median "
              f"{100 * h['near_agreement_median']:.1f}%, p90 {100 * h['near_agreement_p90']:.1f}%; "
              f"coverage {100 * h['raw_coverage']:.0f}% -> {100 * h['supervised']:.0f}%", flush=True)
    return stats


if __name__ == "__main__":
    main(parse())
