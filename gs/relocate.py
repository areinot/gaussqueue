"""Re-place glitch photos: photos whose own LiDAR lands in space the other photos see through.

  python relocate.py --model <sparse/0> --database <matches.db> --depth-dir <sensor_data>
                     --out-model <dir> --out <relocate.json> [--depth-scale S]

ARKit now and then puts a single photo several centimetres off (and smoothness then drags its
neighbours a little with it). Such a photo shows itself through its LiDAR: placed with its pose,
its depth points float in front of surfaces the other photos see behind them. Per photo:
  1. LiDAR check: its high-confidence depth points (scaled by --depth-scale) projected into every
     other photo; a point is "in free space" when at least 3 photos see past it (their depth > 3 cm
     further) and more see past it than agree (within 2 cm). Suspects: free-space share of at
     least --suspect, or of --suspect-low when that's --outlier-ratio times its neighbours' median
     (one glitched frame stands out from the photos taken just before and after it).
  2. Each suspect is solved again against the trusted photos alone: their verified matches are
     triangulated with their poses (two views, < 1.5 px, rays > 2 deg), the suspect's raw matches
     to them give 2D-3D pairs, and pycolmap's LO-RANSAC + refinement solves its pose, everything
     else fixed.
  3. The new pose is kept only if its own LiDAR then agrees better: free-space share below 0.7 of
     before and a larger share agreeing with the others. Otherwise the photo keeps its pose.
Writes the model (images.txt with the accepted poses; cameras.txt, points3D.txt copied) and a
per-photo log of what was checked, tried and done.
"""
import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import box as boxmod
import gsdata

PAIR = 2147483647


def load_lidar(images, depth_dir, scale, dev, stride=2):
    """Per photo: (world-from-camera-ready points in its camera frame [n,3]) and the stacked depth
    and confidence maps with each photo's LiDAR intrinsics, for projecting into it."""
    pts, D, Cf, K = [], [], [], []
    for im in images:
        stem = Path(im["name"]).stem
        mf = Path(depth_dir) / f"{stem}.json"
        d, c = gsdata.read_depth(depth_dir, stem)
        meta = json.loads(mf.read_text()).get("metadata", {}).get("camera") if mf.exists() else None
        if d is None or not meta or "intrinsics" not in meta:
            pts.append(None)
            D.append(None)
            Cf.append(None)
            K.append(None)
            continue
        q = meta["intrinsics"]
        sc = meta["imageResolution"]["width"] / d.shape[1]
        d = d * scale
        h, w = d.shape
        v, u = torch.meshgrid(torch.arange(0, h, stride), torch.arange(0, w, stride), indexing="ij")
        x, y = (u + 0.5) * sc - 0.5, (v + 0.5) * sc - 0.5
        z = d[v, u]
        ok = (c[v, u] == 2) & (z > 0.1)
        pts.append(torch.stack([(x - q[6]) / q[0] * z, (y - q[7]) / q[4] * z, z], -1)[ok].to(dev))
        D.append(d)
        Cf.append(c)
        K.append([q[0], q[4], q[6], q[7], sc])
    have = [k for k, d in enumerate(D) if d is not None]
    shape = D[have[0]].shape if have else (192, 256)
    Dm = torch.zeros(len(images), *shape)
    Cm = torch.zeros(len(images), *shape, dtype=torch.uint8)
    for k in have:
        if D[k].shape == shape:
            Dm[k], Cm[k] = D[k], Cf[k]
    Km = torch.tensor([k if k else [1, 1, 0, 0, 1] for k in K], dtype=torch.float32)
    return pts, Dm.to(dev), Cm.to(dev), Km.to(dev)


def lidar_check(k, vm_k, pts, VM, Dm, Cm, Km, chunk=128):
    """(free-space share, agreeing share) of photo k's LiDAR placed with view matrix vm_k."""
    P = pts[k]
    if P is None or len(P) < 50:
        return None, None
    X = (P - vm_k[:3, 3]) @ vm_k[:3, :3]   # camera -> world
    h, w = Dm.shape[1:]
    th = torch.zeros(len(X), dtype=torch.int32, device=X.device)
    ag = torch.zeros_like(th)
    for s in range(0, len(VM), chunk):
        R, t, Kc = VM[s:s + chunk, :3, :3], VM[s:s + chunk, :3, 3], Km[s:s + chunk]
        pc = torch.einsum("mij,nj->mni", R, X) + t[:, None]
        z = pc[..., 2]
        u = ((Kc[:, 0, None] * pc[..., 0] / z.clamp_min(1e-6) + Kc[:, 2, None] + 0.5) / Kc[:, 4, None] - 0.5).round().long()
        v = ((Kc[:, 1, None] * pc[..., 1] / z.clamp_min(1e-6) + Kc[:, 3, None] + 0.5) / Kc[:, 4, None] - 0.5).round().long()
        inb = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        mi = torch.arange(s, s + len(R), device=X.device)[:, None].expand_as(u)
        inb &= mi != k
        flat = (mi * h * w + v.clamp(0, h - 1) * w + u.clamp(0, w - 1))
        dd = Dm.reshape(-1)[flat]
        good = inb & (Cm.reshape(-1)[flat] == 2) & (dd > 0.1)
        th += (good & (dd > z + 0.03)).sum(0, dtype=torch.int32)
        ag += (good & ((dd - z).abs() < 0.02)).sum(0, dtype=torch.int32)
    fl = (th >= 3) & (th > ag)
    return float(fl.float().mean()), float((ag >= 3).float().mean())


def solve_poses(images, database, suspects, min_pairs=30, max_error=4.0):
    """New view matrices for the suspects, from trusted photos' triangulated matches (pycolmap)."""
    import pycolmap
    by = {Path(im["name"]).stem: k for k, im in enumerate(images)}
    c2w = [np.linalg.inv(im["viewmat"]) for im in images]
    S = set(suspects)
    con = sqlite3.connect(database)
    iname = {i: Path(n).stem for i, n in con.execute("select image_id, name from images")}
    kps = {iname[i]: np.frombuffer(d, np.float32).reshape(r, c)[:, :2]
           for i, r, c, d in con.execute("select image_id, rows, cols, data from keypoints") if d}

    def ray(k, x):
        cam = images[k]["cam"]
        d = np.stack([(x[:, 0] - cam["cx"]) / cam["fx"], (x[:, 1] - cam["cy"]) / cam["fy"], np.ones(len(x))], 1) @ c2w[k][:3, :3].T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def proj(k, X, vm=None):
        cam = images[k]["cam"]
        vm = images[k]["viewmat"] if vm is None else vm
        pc = X @ vm[:3, :3].T + vm[:3, 3]
        return np.stack([cam["fx"] * pc[:, 0] / pc[:, 2] + cam["cx"], cam["fy"] * pc[:, 1] / pc[:, 2] + cam["cy"]], 1), pc[:, 2]

    def pair(pid):
        b = pid % PAIR
        return iname.get((pid - b) // PAIR), iname.get(b)

    pts = defaultdict(list)   # (trusted photo, keypoint) -> 3D points
    for pid, r, _, data in con.execute("select pair_id, rows, cols, data from two_view_geometries where rows > 0"):
        na, nb = pair(pid)
        if na not in by or nb not in by or na not in kps or nb not in kps:
            continue
        ka, kb = by[na], by[nb]
        if ka in S or kb in S:
            continue
        m = np.frombuffer(data, np.uint32).reshape(r, 2)
        xa, xb = kps[na][m[:, 0]], kps[nb][m[:, 1]]
        da, db_ = ray(ka, xa), ray(kb, xb)
        Ca, Cb = c2w[ka][:3, 3], c2w[kb][:3, 3]
        w0 = Ca - Cb
        bb = (da * db_).sum(1)
        den = np.maximum(1 - bb * bb, 1e-9)
        s1 = (bb * (db_ @ w0) - da @ w0) / den
        s2 = ((db_ @ w0) - bb * (da @ w0)) / den
        X = 0.5 * (Ca + s1[:, None] * da + Cb + s2[:, None] * db_)
        ok = (s1 > 0.05) & (s2 > 0.05) & (np.degrees(np.arccos(np.clip(bb, -1, 1))) > 2)
        for k, x in ((ka, xa), (kb, xb)):
            u, z = proj(k, X)
            ok &= (np.hypot(*(u - x).T) < 1.5) & (z > 0)
        for i in np.flatnonzero(ok):
            pts[(ka, int(m[i, 0]))].append(X[i])
            pts[(kb, int(m[i, 1]))].append(X[i])
    P3 = {key: np.median(np.array(v), 0) for key, v in pts.items()}
    pairs = defaultdict(lambda: defaultdict(list))   # suspect -> its keypoint -> 3D points
    for pid, r, _, data in con.execute("select pair_id, rows, cols, data from matches where rows > 0"):
        na, nb = pair(pid)
        if na not in by or nb not in by:
            continue
        ka, kb = by[na], by[nb]
        if (ka in S) == (kb in S):
            continue
        m = np.frombuffer(data, np.uint32).reshape(r, 2)
        f, t, fi, ti = (ka, kb, m[:, 0], m[:, 1]) if ka in S else (kb, ka, m[:, 1], m[:, 0])
        for i, j in zip(fi.tolist(), ti.tolist()):
            X = P3.get((t, j))
            if X is not None:
                pairs[f][i].append(X)
    out = {}
    for f in suspects:
        name, cam = images[f]["name"], images[f]["cam"]
        items = pairs.get(f, {})
        stem = Path(name).stem
        if len(items) < min_pairs or stem not in kps:
            out[f] = {"solved": False, "pairs": len(items), "why": "too few matches to trusted photos"}
            continue
        ii = np.array(sorted(items))
        x2 = kps[stem][ii].astype(np.float64)
        X3 = np.array([np.median(np.array(items[i]), 0) for i in ii])
        camera = pycolmap.Camera(model="PINHOLE", width=int(cam["w"]), height=int(cam["h"]),
                                 params=[cam["fx"], cam["fy"], cam["cx"], cam["cy"]])
        eo = pycolmap.AbsolutePoseEstimationOptions()
        eo.ransac.max_error = max_error
        r_ = pycolmap.estimate_and_refine_absolute_pose(x2, X3, camera, eo)
        if r_ is None:
            out[f] = {"solved": False, "pairs": int(len(ii)), "why": "RANSAC found no pose"}
            continue
        T = r_["cam_from_world"]
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = np.array(T.rotation.matrix()), np.array(T.translation)
        inl = np.array(r_["inlier_mask"])
        u_old, _ = proj(f, X3)
        u_new, _ = proj(f, X3, vm)
        C_old, C_new = c2w[f][:3, 3], -vm[:3, :3].T @ vm[:3, 3]
        turn = np.degrees(np.arccos(np.clip((np.trace(vm[:3, :3] @ images[f]["viewmat"][:3, :3].T) - 1) / 2, -1, 1)))
        out[f] = {"solved": True, "pairs": int(len(ii)), "inliers": int(inl.sum()), "viewmat": vm,
                  "error_px_before": round(float(np.median(np.hypot(*(u_old - x2).T))), 2),
                  "error_px_after": round(float(np.median(np.hypot(*(u_new - x2).T)[inl])), 2),
                  "moved_mm": round(float(np.linalg.norm(C_new - C_old) * 1000), 1), "turn_deg": round(float(turn), 3)}
    return out


def write_model(src, out, images, new_vm):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("cameras.txt", "points3D.txt"):
        if (src / f).exists():
            (out / f).write_bytes((src / f).read_bytes())
    lines = (src / "images.txt").read_text().splitlines()
    res, i = [], 0
    while i < len(lines):
        f = lines[i].split()
        if lines[i].startswith("#") or len(f) < 10:
            res.append(lines[i])
            i += 1
            continue
        if f[9] in new_vm:
            vm = new_vm[f[9]]
            f[1:5] = [f"{x:.12f}" for x in boxmod.rotmat_to_qvec(vm[:3, :3])]
            f[5:8] = [f"{x:.12f}" for x in vm[:3, 3]]
        res += [" ".join(f), lines[i + 1] if i + 1 < len(lines) else ""]
        i += 2
    (out / "images.txt").write_text("\n".join(res) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--database", required=True)
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--out-model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--depth-scale", type=float, help="LiDAR depth scale (default: fitted here, as lidar_scale.py)")
    p.add_argument("--suspect", type=float, default=0.4, help="free-space share that makes a photo a suspect")
    p.add_argument("--suspect-low", type=float, default=0.1,
                   help="... or at least this share and --outlier-ratio times its neighbours' median")
    p.add_argument("--outlier-ratio", type=float, default=4.0)
    p.add_argument("--max-suspects", type=float, default=0.1, help="give up if more than this share of photos are suspects")
    a = p.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    images = gsdata.read_model(Path(a.model))[0]
    scale = a.depth_scale
    if scale is None:
        import lidar_scale
        A = lidar_scale.comparisons(a.model, a.database, a.depth_dir)
        scale = float(1 / np.median(A[:, 1] / A[:, 0])) if len(A) >= 2000 else 1.0
    print(f"{len(images)} photos; LiDAR depth scale {scale:.4f}", flush=True)
    pts, Dm, Cm, Km = load_lidar(images, a.depth_dir, scale, dev)
    VM = torch.tensor(np.array([im["viewmat"] for im in images]), dtype=torch.float32, device=dev)
    check = {}
    for k in range(len(images)):
        check[k] = lidar_check(k, VM[k], pts, VM, Dm, Cm, Km)
    shares = {k: v[0] for k, v in check.items() if v[0] is not None}
    f_all = np.array([shares.get(k, np.nan) for k in range(len(images))])
    def stands_out(k):   # well above the photos taken just before and after (capture order = name order)
        nb = np.r_[f_all[max(0, k - 3):k], f_all[k + 1:k + 4]]
        nb = nb[~np.isnan(nb)]
        return len(nb) > 0 and shares[k] > a.outlier_ratio * np.median(nb)
    suspects = sorted(k for k, f in shares.items()
                      if f >= a.suspect or (f >= a.suspect_low and stands_out(k)))
    log = {"depth_scale": round(scale, 5), "photos": len(images), "checked": len(shares),
           "suspect_share": a.suspect, "free_space_median": round(float(np.median(list(shares.values()) or [0])), 4),
           "suspects": {}, "relocated": []}
    print(f"{len(shares)} photos checked; median free-space share {log['free_space_median']:.1%}; "
          f"{len(suspects)} suspects (>= {a.suspect:.0%}, or >= {a.suspect_low:.0%} and "
          f"{a.outlier_ratio:g}x their neighbours)", flush=True)
    new_vm = {}
    if len(suspects) > a.max_suspects * len(images):
        log["skipped"] = f"{len(suspects)} suspects is too many to trust the rest; poses left as they are"
        print(log["skipped"], flush=True)
    elif suspects:
        sol = solve_poses(images, a.database, suspects)
        for k in suspects:
            name = images[k]["name"]
            r = sol[k]
            entry = {"free_space": round(check[k][0], 3), "agree": round(check[k][1], 3)}
            entry.update({kk: v for kk, v in r.items() if kk != "viewmat"})
            if r["solved"]:
                f1, a1 = lidar_check(k, torch.tensor(r["viewmat"], dtype=torch.float32, device=dev), pts, VM, Dm, Cm, Km)
                entry.update(free_space_after=round(f1, 3), agree_after=round(a1, 3))
                entry["accepted"] = bool(f1 < 0.7 * check[k][0] and a1 > check[k][1])
                if not entry["accepted"]:
                    entry["why"] = "its LiDAR didn't agree better at the new pose"
            else:
                entry["accepted"] = False
            if entry["accepted"]:
                new_vm[name] = r["viewmat"]
                log["relocated"].append(name)
            log["suspects"][name] = entry
            print(f"{Path(name).stem}: free space {entry['free_space']:.0%}"
                  + (f" -> {entry['free_space_after']:.0%}, agree {entry['agree']:.0%} -> {entry['agree_after']:.0%}, "
                     f"moved {entry['moved_mm']:.0f} mm / {entry['turn_deg']:.2f} deg" if r["solved"] else "")
                  + f": {'RELOCATED' if entry['accepted'] else 'kept (' + entry.get('why', '') + ')'}", flush=True)
    write_model(a.model, a.out_model, images, new_vm)
    Path(a.out).write_text(json.dumps(log, indent=1))
    print(f"relocated {len(new_vm)} of {len(suspects)} suspects; model written to {a.out_model}", flush=True)


if __name__ == "__main__":
    main()
