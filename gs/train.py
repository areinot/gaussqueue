"""gsplat trainer for GaussQueue's CUDA workers.

  python train.py <data> --out <dir> [--steps 30000] [--max-res 1920] [--sh-degree 3]
                  [--strategy default|absgrad|mcmc] [--max-splats N]
                  [--depth-dir <sensor_data>] [--depth-weight 0.1] [--depth-conf 2]
                  [--flat-weight 100] [--mode 3dgs|2dgs] [--antialiased]
                  [--init-ply <ply>] [--holdout-every 0] [--export-every N] [--eval-every N]

<data> holds images/ and a COLMAP text model (sparse/0). Output mirrors Brush's layout so
the queue can follow either trainer the same way: <out>/export_<step>.ply checkpoints and
<out>/eval_<step>/<image>.png progress renders, plus <out>/metrics.json at the end.

Depth supervision compares rendered depth with the photo's LiDAR depth only where ARKit's
confidence is at least --depth-conf (2 = high); sky and far geometry have no LiDAR return
(confidence 0) and are left to the photos alone. --flat-weight penalises each splat's
smallest axis so splats lie flat on surfaces. --needle-weight penalises needles: splats
whose longest axis is more than --needle-max times their middle one (flat disks are fine).

Box datasets (box.py) have per-photo masks: colour is trained only on pixels whose surface
lies inside the box, pixels where the box must be empty push transparency (--empty-weight),
the rest are ignored; with --box box.json, splats that leave the box are removed.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
if torch.cuda.is_available():
    from gsplat import rasterization, rasterization_2dgs
    DEVICE = "cuda"
else:   # Apple GPU: metalsplat's Metal kernels behind gsplat's interface (gs/mps_backend.py)
    import mps_backend
    if not mps_backend.available():
        raise SystemExit("train.py needs CUDA (gsplat) or an Apple GPU with metalsplat installed")
    mps_backend.patch_gsplat_strategy()
    rasterization, rasterization_2dgs, DEVICE = mps_backend.rasterization, None, "mps"
from gsplat.strategy import DefaultStrategy, MCMCStrategy

import gsdata

C0 = 0.28209479177387814   # SH band-0 constant


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--max-res", type=int, default=1920)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--strategy", choices=["default", "absgrad", "mcmc", "hybrid", "none"], default="mcmc",
                   help="hybrid: MCMC that clones where absgrad's texture signal is strongest (hybrid.py)")
    p.add_argument("--hybrid-power", type=float, default=1.0, help="hybrid: weight of the absgrad signal")
    p.add_argument("--max-splats", type=int, default=3_000_000)
    p.add_argument("--depth-dir")
    p.add_argument("--eval-depth-dir", help="depth maps for scoring held-out photos (default: --depth-dir)")
    p.add_argument("--depth-weight", type=float, default=0.0)
    p.add_argument("--depth-conf", type=int, default=2)
    p.add_argument("--depth-scale", type=float, default=1.0,
                   help="multiply LiDAR depth by this (training and evaluation): iPhone LiDAR can read a few %% long "
                        "against the photos' scale")
    p.add_argument("--depth-align-after", type=float, default=0.0,
                   help="from this fraction of training on, align each photo's LiDAR depth to the "
                        "rendered surface (per-photo scale + offset) before comparing: LiDAR keeps "
                        "the shape, the photos decide where it sits (0 = off)")
    p.add_argument("--depth-deadband", type=float, default=0.0,
                   help="metres of depth disagreement that cost nothing (LiDAR is good to ~1 cm)")
    p.add_argument("--flat-weight", type=float, default=0.0)
    p.add_argument("--mode", choices=["3dgs", "2dgs"], default="3dgs")
    p.add_argument("--space", help="space.npz from fuse.py --grid-only: known-empty / surface / unknown")
    p.add_argument("--carve", action="store_true",
                   help="every 100 steps, remove splats inside known-empty space and shorten ones poking into it")
    p.add_argument("--carve-fade", type=float, default=0.0,
                   help="soft carve: multiply opacity by this per pass instead of removing (e.g. 0.5); "
                        "splats the photos need (glass LiDAR sees through) regrow, floaters fade out")
    p.add_argument("--carve-band", type=float, default=0.0,
                   help="only carve within this many metres of a surface LiDAR measured (spikes poking out); "
                        "empty space far from any surface (where glasses stand) is left alone (0 = everywhere)")
    p.add_argument("--far-flat-weight", type=float, default=0.0,
                   help="make splats outside the scanned region thin along the line to its centre (billboards)")
    p.add_argument("--antialiased", action="store_true")
    p.add_argument("--box", help="box.json from box.py: keep splats inside the box")
    p.add_argument("--bounds", default="auto",
                   help="where splats may live: auto (a level box around the cameras and their LiDAR points, "
                        "padded by --bounds-pad), off, or a box.json; splats outside are moved back in (MCMC) "
                        "and dropped from the result")
    p.add_argument("--bounds-pad", type=float, default=1.0,
                   help="auto bounds: metres added on every side, and at least a quarter of the box's size")
    p.add_argument("--open-scene", type=float, default=0.6,
                   help="auto bounds: below this share of photo pixels with a LiDAR return (sky, far views) the "
                        "scene is open: --sky-points are kept and the bounds grow to hold them; above it they're dropped")
    p.add_argument("--relocate-until", type=float, default=0.97,
                   help="MCMC: keep moving invisible splats to where they are needed until this fraction of "
                        "training (adding new ones still stops at the refine stop)")
    p.add_argument("--keep-invisible", action="store_true",
                   help="keep splats too faint to show (opacity under 1/255) in the result")
    p.add_argument("--empty-weight", type=float, default=0.1,
                   help="masked datasets: weight of rendered opacity where the box must be empty")
    p.add_argument("--sh-ramp", type=int, default=1000,
                   help="steps per added SH degree (colour detail waits for the shapes; default 1000)")
    p.add_argument("--opacity-reg", type=float, default=0.01, help="MCMC: pull opacities down (default 0.01)")
    p.add_argument("--scale-reg", type=float, default=0.01, help="MCMC: pull sizes down (default 0.01)")
    p.add_argument("--opacity-binary", type=float, default=0.0,
                   help="weight pushing each splat to fully opaque or fully clear (solid surfaces, no haze)")
    p.add_argument("--opacity-binary-start", type=float, default=0.6,
                   help="fraction of the run after which --opacity-binary applies (default 0.6)")
    p.add_argument("--noise-stop", type=float, default=0.0,
                   help="MCMC: fraction of the run after which splats stop being jittered (0 = never)")
    p.add_argument("--depth-front-only", action="store_true",
                   help="depth loss only where the render is in front of the LiDAR surface (spikes poking "
                        "out); behind it (reflections in metal, glass) is free")
    p.add_argument("--streak-weight", type=float, default=0.0,
                   help="penalise splats that look like streaks on screen (a needle, or a flat disk seen "
                        "edge-on): projected length over --streak-max times the width")
    p.add_argument("--streak-max", type=float, default=10.0)
    p.add_argument("--streak-off-surface", action="store_true",
                   help="with --space: only penalise streaks away from LiDAR surfaces, so flat splats "
                        "lying on a surface seen at a grazing angle are left alone")
    p.add_argument("--streak-pass-every", type=int, default=0,
                   help="every N steps, render the training views and fade splats that look like streaks "
                        "(edge-on disks, needles) in most of the views that see them (0 = off)")
    p.add_argument("--streak-pass-share", type=float, default=0.5,
                   help="fade a splat when it is a streak in more than this share of the views that see it")
    p.add_argument("--streak-pass-fade", type=float, default=0.5, help="opacity multiplier per pass")
    p.add_argument("--needle-weight", type=float, default=0.0)
    p.add_argument("--needle-max", type=float, default=5.0)
    p.add_argument("--init-ply")
    p.add_argument("--points", help="points3D.txt to initialise from instead of the model's")
    p.add_argument("--sky-points", type=int, default=0,
                   help="also start with this many splats on a distant sphere (sky, far scenery)")
    p.add_argument("--means-lr", type=float, default=1.6e-4, help="times scene scale")
    p.add_argument("--lr-decay-steps", type=int, default=0,
                   help="decay the position learning rate over this many steps, then hold it "
                        "(0 = over the whole run, the old behaviour)")
    p.add_argument("--lr-final", type=float, default=0.01,
                   help="position learning rate at the end of the decay, as a fraction of the start")
    p.add_argument("--refine-stop", type=int, default=0,
                   help="stop densifying at this step (0 = 85%% of the run for MCMC, half for default/absgrad)")
    p.add_argument("--pose-opt", action="store_true",
                   help="fine-tune each training photo's camera pose (small shift + rotation) against "
                        "the splats; the corrections are reported in metrics.json")
    p.add_argument("--pose-lr", type=float, default=1e-5, help="--pose-opt learning rate")
    p.add_argument("--pose-reg", type=float, default=1e-6, help="--pose-opt pull towards no correction")
    p.add_argument("--sweep-correction", action="store_true",
                   help="sweep correction: also fine-tune one rigid correction per sweep (a run of photos "
                        "taken in one go), moving all its cameras together. Fixes sweeps that are shifted "
                        "wholesale against the rest (ARKit drift), which per-photo --pose-opt can't")
    p.add_argument("--sweep-lr", type=float, default=3e-5, help="sweep correction learning rate")
    p.add_argument("--sweep-start", type=int, default=500, help="start sweep correction at this step")
    p.add_argument("--sweep-jump", type=float, default=0.0,
                   help="a new sweep starts where the camera jumps further than this (m) between shots "
                        "(0 = 3x the capture's median step, at least 5 cm)")
    p.add_argument("--sweep-turn", type=float, default=25.0,
                   help="... or turns more than this (degrees) between shots")
    p.add_argument("--sweep-max", type=int, default=0, help="split sweeps longer than this many photos (0 = never)")
    p.add_argument("--track-cameras", help="a COLMAP folder of extra cameras (e.g. novel angles) rendered "
                                           "at every --test-every step into <out>/track")
    p.add_argument("--holdout-every", type=int, default=0)
    p.add_argument("--holdout-names", help="file listing the photos to hold out (one name per line), "
                                           "so runs on different datasets are scored on the same views")
    p.add_argument("--subsample", type=int, default=1, help="train on every Nth photo")
    p.add_argument("--exclude-names", help="file listing photos not to train on (one name per line)")
    p.add_argument("--photo-scales", help="json {photo: factor <= 1}: train those photos at a lower resolution "
                                          "(frame checks: a blurred photo kept at the size where its blur is a normal edge)")
    p.add_argument("--pixel-weights", help="folder of per-photo weight maps (coverage.py): pixels that "
                                           "close-ups cover better count less")
    p.add_argument("--pixel-weights-absolute", action="store_true",
                   help="use --pixel-weights as absolute loss weights (scaled by the folder's "
                        "summary.json scale), so they also shift weight between photos; by default "
                        "they only redistribute it within each photo")
    p.add_argument("--view-weights", help="JSON {photo name: weight}: photos are picked for training in "
                                          "proportion to their weight (e.g. how densely they sample the "
                                          "surface, so close-ups lead); missing photos weigh 1")
    p.add_argument("--first-names", help="file listing the only photos used until --first-until "
                                         "(e.g. the distant ones: coarse shape and colour first)")
    p.add_argument("--first-until", type=float, default=0.0, help="fraction of training for --first-names")
    p.add_argument("--ssim-weight", type=float, default=0.2,
                   help="share of the photo loss given to SSIM (edges, texture); the rest is L1")
    p.add_argument("--export-every", type=int, default=0)
    p.add_argument("--test-every", type=int, default=0,
                   help="score the held-out photos every N steps into <out>/curve.json (quality vs steps)")
    p.add_argument("--track-views", nargs="*", default=[],
                   help="held-out photos whose renders are saved at every --test-every step")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# ---------------------------------------------------------------- model

def knn_scale(means, k=3, chunk=1024):
    """Mean distance to the k nearest neighbours (initial splat size)."""
    out = torch.empty(means.shape[0], device=means.device)
    for i in range(0, means.shape[0], chunk):
        d = torch.cdist(means[i:i + chunk], means)
        out[i:i + chunk] = d.topk(k + 1, largest=False).values[:, 1:].mean(-1)
    return out.clamp_min(1e-7)


def init_from_points(pts, rgb, sh_degree, init_opacity, init_scale, device):
    means = torch.from_numpy(pts).float().to(device)
    n = means.shape[0]
    scales = torch.log(knn_scale(means) * init_scale)[:, None].repeat(1, 3)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1
    k = (sh_degree + 1) ** 2 - 1
    return {"means": means, "quats": quats, "scales": scales,
            "opacities": torch.logit(torch.full((n,), init_opacity, device=device)),
            "sh0": ((torch.from_numpy(rgb).float().to(device) - 0.5) / C0)[:, None, :],
            "shN": torch.zeros(n, k, 3, device=device)}


def sky_shell(n, center, radius):
    """n points spread evenly over a sphere (Fibonacci lattice), light grey."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    d = np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1)
    return (center + radius * d).astype(np.float32), np.full((n, 3), 0.8, dtype=np.float32)


class Space:
    """The LiDAR's map of space (fuse.py): 0 unknown, 1 known empty, 2 surface."""

    def __init__(self, path, dev):
        z = np.load(path)
        self.grid = torch.from_numpy(z["grid"]).to(dev)
        self.origin = torch.tensor(z["origin"], dtype=torch.float32, device=dev)
        self.voxel = float(z["voxel"])
        nz, ny, nx = self.grid.shape
        self.n = torch.tensor([nx, ny, nz], device=dev)
        self.center = self.origin + self.n * self.voxel / 2

    def near_surface(self, P, band):
        """Whether points lie within `band` metres of a surface cell."""
        k = max(1, int(round(band / self.voxel)))
        if getattr(self, "_near", None) is None or self._near[0] != k:
            surf = (self.grid == 2).float()[None, None]
            self._near = (k, F.max_pool3d(surf, 2 * k + 1, stride=1, padding=k)[0, 0] > 0)
        idx = torch.floor((P - self.origin) / self.voxel).long()
        inside = ((idx >= 0) & (idx < self.n)).all(-1)
        idx = torch.minimum(idx.clamp(min=0), self.n - 1)
        return inside & self._near[1][idx[:, 2], idx[:, 1], idx[:, 0]]

    def lookup(self, P):
        idx = torch.floor((P - self.origin) / self.voxel).long()
        inside = ((idx >= 0) & (idx < self.n)).all(-1)
        idx = torch.minimum(idx.clamp(min=0), self.n - 1)
        st = self.grid[idx[:, 2], idx[:, 1], idx[:, 0]]
        return torch.where(inside, st, torch.zeros_like(st)), inside


def quat_to_rotmat(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


@torch.no_grad()
def carve(params, space, min_logit=-6.0, fade=0.0, band=0.0):
    """Remove splats centred in known-empty space (opacity to ~0: MCMC relocates them, default
    densification prunes them) and shorten long axes whose tips poke into it."""
    means = params["means"]
    st, _ = space.lookup(means)
    dead = st == 1
    if band > 0:
        dead &= space.near_surface(means, band)
    if fade > 0:   # soft: fade, and let the photos decide
        o = torch.sigmoid(params["opacities"][dead]) * fade
        params["opacities"][dead] = torch.logit(o.clamp(1e-6, 1 - 1e-6))
    else:
        params["opacities"][dead] = min_logit
    s = torch.exp(params["scales"])
    k = s.argmax(-1)
    R = quat_to_rotmat(params["quats"])
    axis = R[torch.arange(len(k), device=k.device), :, k]
    smax = s.gather(1, k[:, None])[:, 0]
    alive = ~dead
    shrunk = 0
    for frac in (1.0, 0.5):   # tip at 2 sigma; halve until it no longer reaches empty space
        reach = 2 * smax * frac
        poke = alive & ((space.lookup(means + reach[:, None] * axis)[0] == 1)
                        | (space.lookup(means - reach[:, None] * axis)[0] == 1))
        if band > 0:
            poke &= space.near_surface(means, band)
        if not poke.any():
            break
        new = torch.log(smax[poke] * 0.5)
        params["scales"][poke, k[poke]] = torch.minimum(params["scales"][poke, k[poke]], new)
        smax[poke] *= 0.5
        shrunk += int(poke.sum())
    return int(dead.sum()), shrunk


def far_flat_loss(params, space):
    """Splats outside the scanned region: penalise their extent along the line to its centre,
    so distant scenery becomes camera-facing disks instead of needles reaching into the scene."""
    means = params["means"]
    _, inside = space.lookup(means.detach())
    far = ~inside
    if not far.any():
        return means.new_zeros(())
    s = torch.exp(params["scales"][far])
    R = quat_to_rotmat(params["quats"][far])
    r = F.normalize(means[far].detach() - space.center, dim=-1)
    along = (R.transpose(1, 2) @ r[:, :, None])[:, :, 0]   # r in each splat's local axes
    sigma_r = ((s * along) ** 2).sum(-1).clamp_min(1e-12).sqrt()
    return (sigma_r / s.max(-1).values).mean()


class Box:
    def __init__(self, path, dev):
        b = path if isinstance(path, dict) else json.loads(Path(path).read_text())
        self.center = torch.tensor(b["center"], dtype=torch.float32, device=dev)
        self.axes = torch.tensor(b["axes"], dtype=torch.float32, device=dev)
        self.half = torch.tensor(b["half"], dtype=torch.float32, device=dev)

    def inside(self, P, pad=0.02):
        return (((P - self.center) @ self.axes).abs() <= self.half * (1 + pad)).all(-1)


def auto_bounds(images, depth_dir, depth_scale=1.0, pad=1.0, conf=2, max_depth=6.0, every=3, dev="cpu"):
    """A level box (world +y up, as ARKit) around the camera centres and their high-confidence
    LiDAR points, turned about +y to the points' main horizontal axes; per axis it spans the
    0.5-99.5th percentile of the points (a few LiDAR strays don't stretch it) and all the cameras,
    then grows by max(pad, a quarter of its size) on every side. None without any points."""
    pts, cams, cover = [], [], []
    for k, im in enumerate(images):
        c2w = np.linalg.inv(im["viewmat"])
        cams.append(c2w[:3, 3])
        if not depth_dir or k % every:
            continue
        d, c = gsdata.read_depth(depth_dir, im["name"])
        if d is None:
            continue
        d, c = d.numpy() * depth_scale, c.numpy()
        h, w = d.shape
        cam = im["cam"]
        cover.append(float(((c >= 1) & (d > 0.05) & (d < max_depth)).mean()))
        v, u = np.mgrid[0:h:2, 0:w:2]
        z = d[v, u]
        ok = (c[v, u] >= conf) & (z > 0.05) & (z < max_depth)
        z, u, v = z[ok], (u[ok] + 0.5) * cam["w"] / w, (v[ok] + 0.5) * cam["h"] / h
        X = np.stack([(u - cam["cx"]) / cam["fx"] * z, (v - cam["cy"]) / cam["fy"] * z, z], 1)
        pts.append(X @ c2w[:3, :3].T + c2w[:3, 3])
    cams = np.array(cams)
    if not pts:
        return None
    P = np.concatenate(pts)
    xz = P[:, [0, 2]] - np.median(P[:, [0, 2]], 0)
    ev, vec = np.linalg.eigh(xz.T @ xz)
    a1 = np.array([vec[0, 1], 0, vec[1, 1]])   # main horizontal axis
    axes = np.stack([a1, [0, 1, 0], np.cross(a1, [0, 1, 0])], 1)   # columns: box axes in world
    pl, cl = P @ axes, cams @ axes
    lo = np.minimum(np.percentile(pl, 0.5, 0), cl.min(0))
    hi = np.maximum(np.percentile(pl, 99.5, 0), cl.max(0))
    grow = np.maximum(pad, 0.25 * (hi - lo))
    lo, hi = lo - grow, hi + grow
    return {"center": ((lo + hi) / 2 @ axes.T).tolist(), "axes": axes.tolist(),
            "half": ((hi - lo) / 2).tolist(), "lidar_points": int(len(P)), "cameras": int(len(cams)),
            "lidar_coverage": round(float(np.mean(cover)), 3)}


@torch.no_grad()
def keep_in_box(params, box, min_logit=-6.0):
    out = ~box.inside(params["means"])
    params["opacities"][out] = min_logit   # MCMC relocates them, default densification prunes
    return int(out.sum())


def screen_shapes(meta):
    """Per projected splat in a render: (length/width on screen, splat index, opacity, length px).
    From gsplat's 2D conics (inverse screen covariance), packed or not."""
    con = meta["conics"].reshape(-1, 3)
    n = con.shape[0]
    a, b, c = con.unbind(-1)
    half, disc = (a + c) / 2, (((a - c) / 2) ** 2 + b * b).sqrt()
    l1, l2 = half + disc, (half - disc).clamp_min(1e-12)        # conic eigenvalues: 1/sigma^2
    aspect = (l1 / l2).clamp_min(1.0).sqrt()
    radii = meta["radii"].reshape(n, -1)
    ids = meta.get("gaussian_ids")
    ids = ids.reshape(-1) if ids is not None else torch.arange(n, device=con.device)
    vis = (radii > 0).all(-1)
    return aspect[vis], ids[vis], meta["opacities"].reshape(-1)[vis], l2[vis].rsqrt()


def streak_loss(meta, means, ratio, space=None):
    """Projected streaks (long, thin on screen and opaque enough to show) cost by how far their
    length/width exceeds ratio, weighted by opacity; with space, splats in surface cells are free."""
    aspect, ids, opac, length = screen_shapes(meta)
    w = opac * (length > 2.0)
    if space is not None:
        st, _ = space.lookup(means[ids].detach())
        w = w * (st != 2)
    return (w * F.relu(aspect.log() - math.log(ratio))).sum() / w.sum().clamp_min(1.0)


def streak_share(meta):
    """Share of visible splats (opacity > 0.1, over 2 px long) that are streaks on screen (10:1)."""
    aspect, _, opac, length = screen_shapes(meta)
    keep = (opac > 0.1) & (length > 2.0)
    return float((aspect[keep] > 10).float().mean()) if keep.any() else None


@torch.no_grad()
def streak_pass(params, views, sh_degree, mode, antialiased, share, fade, ratio=10.0, space=None,
                viewmat=None):
    """Fade splats that are streaks on screen (projected over ratio:1, over 2 px long, opacity
    over 0.1) in more than `share` of the training views that see them, by `fade`. Splats needed
    by the photos regrow their opacity; the rest fade out (MCMC then reuses them). With space,
    splats in LiDAR surface cells are spared. Returns the number faded."""
    n = len(params["means"])
    dev = params["means"].device
    seen = torch.zeros(n, device=dev)
    streak = torch.zeros(n, device=dev)
    for i in range(len(views)):
        vm = viewmat(views.viewmats[i], i) if viewmat is not None else None
        *_, meta, _ = render(params, views, i, sh_degree, mode, antialiased, viewmat=vm)
        aspect, ids, opac, length = screen_shapes(meta)
        keep = (opac > 0.1) & (length > 2.0)
        seen.index_add_(0, ids[keep], torch.ones_like(aspect[keep]))
        streak.index_add_(0, ids[keep], (aspect[keep] > ratio).float())
    bad = (seen >= 3) & (streak > share * seen)
    if space is not None:
        bad &= space.lookup(params["means"])[0] != 2
    if bad.any():
        o = torch.sigmoid(params["opacities"][bad]) * fade
        params["opacities"][bad] = torch.logit(o.clamp(1e-6, 1 - 1e-6))
    return int(bad.sum())


def needle_loss(params, ratio):
    s = torch.exp(params["scales"]).sort(-1).values   # ascending: min, mid, max
    return F.relu(s[:, 2] / s[:, 1].clamp_min(1e-9) - ratio).mean()


@torch.no_grad()
def shape_stats(params):
    """Share of visible splats (opacity > 0.1) that are needles (longest axis > 10x the middle)
    and that are flat (smallest axis < 0.1x the middle)."""
    vis = torch.sigmoid(params["opacities"]) > 0.1
    s = torch.exp(params["scales"][vis]).sort(-1).values
    if not len(s):
        return {"visible": 0}
    return {"visible": int(vis.sum()),
            "needle_pct": round(100 * float((s[:, 2] > 10 * s[:, 1]).float().mean()), 2),
            "flat_pct": round(100 * float((s[:, 0] < 0.1 * s[:, 1]).float().mean()), 2)}


def axis_angle_to_R(v):
    """Rotation matrices [N,3,3] from axis-angle vectors [N,3] (Rodrigues), differentiable."""
    th = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = v / th
    K = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=v.device, dtype=v.dtype)[None] + s * K + (1 - c) * (K @ K)


def find_sweeps(names, c2w, jump=0.0, turn=25.0, longest=0, min_size=3):
    """Sweep id per photo: photos in capture order (their names), split where the camera jumps
    or turns sharply between shots (a pause, or a new side of the scene). ARKit's drift moves a
    continuous run of shots together, so a sweep is one such run.

    jump: metres; 0 = automatic, 3x this capture's median step between shots (at least 5 cm),
    so a walking capture (a shot every few cm) isn't cut at every step. longest: split runs
    longer than this many photos (0 = never). Runs shorter than min_size photos join the
    neighbouring run they continue most smoothly (a lone photo can't be measured as a sweep)."""
    order = sorted(range(len(names)), key=lambda k: names[k])
    if not order:
        return []
    pos, axis = c2w[:, :3, 3], c2w[:, :3, 2]
    step = [float((pos[a] - pos[b]).norm()) for a, b in zip(order, order[1:])]
    ang = [math.degrees(math.acos(float((axis[a] * axis[b]).sum().clamp(-1, 1)))) for a, b in zip(order, order[1:])]
    if jump <= 0:
        jump = max(0.05, 3 * sorted(step)[len(step) // 2]) if step else 0.05
    runs, cur = [], [order[0]]
    for k, b in enumerate(order[1:]):
        if step[k] > jump or ang[k] > turn or (longest and len(cur) >= longest):
            runs.append(cur)
            cur = []
        cur.append(b)
    runs.append(cur)
    # fold short runs into a neighbour: the side with the smaller jump (scaled by turn)
    cost = lambda a, b: float((pos[a] - pos[b]).norm()) / jump + math.degrees(
        math.acos(float((axis[a] * axis[b]).sum().clamp(-1, 1)))) / turn
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, r in enumerate(runs):
            if len(r) >= min_size:
                continue
            left = cost(runs[i - 1][-1], r[0]) if i > 0 else float("inf")
            right = cost(r[-1], runs[i + 1][0]) if i + 1 < len(runs) else float("inf")
            if left <= right:
                runs[i - 1] += r
            else:
                runs[i + 1] = r + runs[i + 1]
            del runs[i]
            changed = True
            break
    sid = [0] * len(names)
    for i, r in enumerate(runs):
        for k in r:
            sid[k] = i
    return sid


class PoseCorrection(torch.nn.Module):
    """Per-photo correction applied in the camera's own frame: camera-to-world becomes
    camera-to-world @ [R(rot) | shift]. Starts at zero.

    With sweeps (sweep correction), each sweep also gets one rigid correction in world space,
    a turn about the sweep's centre plus a shift, applied to all its cameras together:
    camera-to-world becomes G(sweep) @ camera-to-world @ [R(rot) | shift]. The largest sweep
    stays put as the anchor, so the whole scene can't wander."""

    def __init__(self, n, dev, sweeps=None, viewmats=None):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.zeros(n, 3, device=dev))
        self.rot = torch.nn.Parameter(torch.zeros(n, 3, device=dev))
        self.sweeps = None
        if sweeps is not None:
            m = max(sweeps) + 1
            self.sweeps = torch.tensor(sweeps, device=dev)
            centres = torch.linalg.inv(viewmats)[:, :3, 3]
            self.centre = torch.stack([centres[self.sweeps == g].mean(0) for g in range(m)])
            self.anchor = int(torch.bincount(self.sweeps).argmax())
            self.sweep_shift = torch.nn.Parameter(torch.zeros(m, 3, device=dev))
            self.sweep_rot = torch.nn.Parameter(torch.zeros(m, 3, device=dev))

    def per_photo(self):
        return [self.shift, self.rot]

    def per_sweep(self):
        return [self.sweep_shift, self.sweep_rot] if self.sweeps is not None else []

    def viewmat(self, vm, i):
        c2w = torch.linalg.inv(vm)
        R = axis_angle_to_R(self.rot[i:i + 1])[0]
        bottom = torch.tensor([[0., 0, 0, 1]], device=vm.device)
        T = torch.cat([torch.cat([R, self.shift[i][:, None]], 1), bottom], 0)
        if self.sweeps is not None and int(self.sweeps[i]) != self.anchor:
            g = int(self.sweeps[i])
            Rg = axis_angle_to_R(self.sweep_rot[g:g + 1])[0]
            tg = self.centre[g] - Rg @ self.centre[g] + self.sweep_shift[g]
            c2w = torch.cat([torch.cat([Rg, tg[:, None]], 1), bottom], 0) @ c2w
        return torch.linalg.inv(c2w @ T)


def make_params(splats, device):
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device).contiguous()) for k, v in splats.items()})


def render(params, views, idx, sh_degree, mode="3dgs", antialiased=False, absgrad=False,
           packed=True, colors=None, scales=None, means=None, viewmat=None):
    """Render view(s) idx -> (rgb [H,W,3], depth [H,W], alpha [H,W], meta, extras)."""
    w, h = views.size(idx)
    means = params["means"] if means is None else means
    scales = torch.exp(params["scales"]) if scales is None else scales
    opac = torch.sigmoid(params["opacities"])
    sh = colors is None
    if sh:
        colors = torch.cat([params["sh0"], params["shN"]], 1)
    vm, K = (views.viewmats[idx:idx + 1] if viewmat is None else viewmat[None]), views.K[idx:idx + 1]
    if mode == "2dgs":
        out, alpha, normals, surf_normals, distort, _median, meta = rasterization_2dgs(
            means, params["quats"], scales, opac, colors, vm, K, w, h,
            sh_degree=sh_degree if sh else None, render_mode="RGB+ED", absgrad=absgrad,
            distloss=True, packed=False)
        extras = {"normals": normals[0], "surf_normals": surf_normals[0], "distort": distort[0, ..., 0]}
    else:
        out, alpha, meta = rasterization(
            means, params["quats"], scales, opac, colors, vm, K, w, h,
            sh_degree=sh_degree if sh else None, render_mode="RGB+ED", packed=packed, absgrad=absgrad,
            rasterize_mode="antialiased" if antialiased else "classic")
        extras = {}
    out = out[0]
    return out[..., :-1], out[..., -1], alpha[0, ..., 0], meta, extras


@torch.no_grad()
def align_depth(d, gt):
    """Scale and offset taking this photo's LiDAR depth onto the rendered surface: least
    squares, refitted twice without the worst fifth (edges, glass), kept small (LiDAR's
    errors are a centimetre or two, not a different scene)."""
    keep = torch.ones_like(gt, dtype=torch.bool)
    s, b = gt.new_tensor(1.0), gt.new_tensor(0.0)
    for _ in range(3):
        x, y = gt[keep], d[keep]
        if len(x) < 50:
            break
        xm, ym = x.mean(), y.mean()
        var = ((x - xm) ** 2).mean()
        s = (((x - xm) * (y - ym)).mean() / var).clamp(0.97, 1.03) if var > 1e-8 else gt.new_tensor(1.0)
        b = (ym - s * xm).clamp(-0.03, 0.03)
        r = (s * gt + b - d).abs()
        keep = r <= torch.quantile(r[:1_000_000] if r.numel() > 1_000_000 else r, 0.8)
    return s, b


def pool_to(x, size):
    """adaptive_avg_pool2d, also where MPS can't (sizes that don't divide): on the Apple GPU,
    resample to the next exact multiple first, then average blocks."""
    h, w = size
    H, W = x.shape[-2:]
    if x.device.type != "mps" or (H % h == 0 and W % w == 0):
        return F.adaptive_avg_pool2d(x, size)
    kh, kw = -(-H // h), -(-W // w)
    x = F.interpolate(x, size=(h * kh, w * kw), mode="bilinear", align_corners=False)
    return F.avg_pool2d(x, (kh, kw))


def depth_loss(depth, alpha, gt, conf, min_conf, deadband=0.0, align=False, front_only=False):
    """Relative L1 between rendered depth (pooled to the LiDAR map's size) and LiDAR depth,
    only where the LiDAR is confident; disagreement within the deadband (metres) is free.
    With align, the LiDAR depth is first fitted onto the rendered surface (align_depth)."""
    h, w = gt.shape
    d = pool_to(depth[None, None], (h, w))[0, 0]
    a = pool_to(alpha[None, None], (h, w))[0, 0]
    m = (conf >= min_conf) & (gt > 0.05) & torch.isfinite(gt) & (a > 0.5)
    if not m.any():
        return depth.new_zeros(())
    if align and m.sum() > 100:
        sc, off = align_depth(d.detach()[m], gt[m])
        gt = gt * sc + off
    err = (gt[m] - d[m]).clamp_min(0) if front_only else (d[m] - gt[m]).abs()
    return ((err - deadband).clamp_min(0) / gt[m]).mean()


# ---------------------------------------------------------------- training

def depth_error(depth, alpha, gt, conf):
    """Median relative error of rendered depth against confident LiDAR depth."""
    h, w = gt.shape
    d = pool_to(depth[None, None], (h, w))[0, 0]
    m = (conf >= 2) & (gt > 0.05) & torch.isfinite(gt)
    if not m.any():
        return None
    a = pool_to(alpha[None, None], (h, w))[0, 0]
    err = torch.where(a[m] > 0.5, (d[m] - gt[m]).abs() / gt[m], torch.ones_like(gt[m]))
    return float(err.median())


def in_front(depth, alpha, gt, conf, margin=0.02):
    """Floaters and spikes poking out: share of confident LiDAR pixels where the render sits
    more than `margin` metres in front of the surface LiDAR measured."""
    h, w = gt.shape
    d = pool_to(depth[None, None], (h, w))[0, 0]
    a = pool_to(alpha[None, None], (h, w))[0, 0]
    m = (conf >= 2) & (gt > 0.05) & torch.isfinite(gt) & (a > 0.5)
    if not m.any():
        return None
    return float((d[m] < gt[m] - margin).float().mean())


def scene_counts(m, params):
    """' needles 3.1% holes 4.0% in front 1.2%' for a log line: spikes (share of visible splats
    10x longer than wide), gaps (box pixels left see-through), floaters (render in front of LiDAR)."""
    sh = shape_stats(params)
    out = f" needles {sh['needle_pct']:.1f}%" if "needle_pct" in sh else ""
    for k, lab in (("streaks", "streaks"), ("holes", "holes"), ("in_front", "in front")):
        if m.get(k) is not None:
            out += f" {lab} {100 * m[k]:.1f}%"
    return out


def depth_image(depth, alpha):
    """Inverse depth as grey (near = bright), black where nothing was rendered."""
    inv = torch.where(alpha > 0.5, 1 / depth.clamp_min(0.05), torch.zeros_like(depth))
    hi = torch.quantile(inv[inv > 0][:1_000_000], 0.98) if (inv > 0).any() else 1.0
    return (inv / hi).clamp(0, 1)[..., None].repeat(1, 1, 3)


def evaluate(params, views, sh_degree, mode, antialiased, out_dir=None):
    """PSNR/SSIM (and depth error where the views have LiDAR) over views; optionally
    saves the renders and depth maps."""
    res = []
    with torch.no_grad():
        for i in range(len(views)):
            rgb, depth, alpha, meta, _ = render(params, views, i, sh_degree, mode, antialiased)
            gt = views.image(i)
            rgb = rgb.clamp(0, 1)
            m = views.masks[i] == 255 if views.masks and views.masks[i] is not None else None
            r = {"name": views.names[i], "psnr": gsdata.psnr(rgb, gt, m), "ssim": float(gsdata.ssim(rgb, gt, m))}
            try:
                r["streaks"] = streak_share(meta)
            except Exception:
                pass
            if m is not None and m.any():   # gaps: box pixels the splats leave see-through
                r["holes"] = float((alpha[m] < 0.9).float().mean())
            if views.depth[i] is not None:
                r["depth_err"] = depth_error(depth, alpha, views.depth[i], views.conf[i])
                r["in_front"] = in_front(depth, alpha, views.depth[i], views.conf[i])
            res.append(r)
            if out_dir:
                stem = Path(views.names[i]).stem
                gsdata.save_png(Path(out_dir) / f"{stem}.png", rgb)
                gsdata.save_png(Path(out_dir) / f"{stem}_depth.png", depth_image(depth, alpha))
    if not res:
        return {}
    out = {"psnr": sum(r["psnr"] for r in res) / len(res), "ssim": sum(r["ssim"] for r in res) / len(res),
           "views": res}
    for k in ("depth_err", "holes", "in_front", "streaks"):
        v = [r[k] for r in res if r.get(k) is not None]
        if v:
            out[k] = sum(v) / len(v)
    return out


def train(a, splats=None, log=print):
    """Train (or fine-tune, when `splats` or --init-ply is given). Returns (params, metrics)."""
    gsdata.keep_awake()
    torch.manual_seed(a.seed)
    dev = DEVICE
    root = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images, pts, rgb = gsdata.read_model(gsdata.find_model(root))
    if a.holdout_names:
        held = {l.strip() for l in Path(a.holdout_names).read_text().splitlines() if l.strip()}
        te_ims = [im for im in images if im["name"] in held]
        tr_ims = [im for im in images if im["name"] not in held]
    else:
        tr_ims, te_ims = gsdata.split(images, a.holdout_every)
    tr_ims = tr_ims[::max(1, a.subsample)]
    if a.exclude_names:
        skip = {l.strip() for l in Path(a.exclude_names).read_text().splitlines() if l.strip()}
        tr_ims = [im for im in tr_ims if im["name"] not in skip]
    photo_scales = json.loads(Path(a.photo_scales).read_text()) if a.photo_scales and Path(a.photo_scales).exists() else {}
    tv = gsdata.Views(root, tr_ims, a.max_res, dev, a.depth_dir if a.depth_weight > 0 else None,
                      a.pixel_weights, photo_scales)
    if photo_scales:
        log(f"{sum(1 for im in tr_ims if im['name'] in photo_scales)} photos trained at a lower resolution (--photo-scales)")
    pw_scale = 1.0
    if a.pixel_weights and a.pixel_weights_absolute:
        sf = Path(a.pixel_weights) / "summary.json"
        pw_scale = float(json.loads(sf.read_text()).get("scale", 1.0)) if sf.exists() else 1.0
    ev = gsdata.Views(root, te_ims, a.max_res, dev, a.eval_depth_dir or a.depth_dir)
    if a.depth_scale != 1.0:
        for views in (tv, ev):
            views.depth = [None if d is None else d * a.depth_scale for d in views.depth]
    centers = tv.centers()
    scene_scale = float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1 or 1.0
    bounds = None
    if a.bounds == "auto" and not a.box:
        bounds = auto_bounds(images, a.depth_dir or a.eval_depth_dir, a.depth_scale, a.bounds_pad)
        if bounds and a.sky_points:
            if bounds["lidar_coverage"] >= a.open_scene:   # enclosed: nothing distant for a sky to stand for
                log(f"LiDAR returns over {bounds['lidar_coverage']:.0%} of the photos: enclosed scene, no sky splats")
                a.sky_points = 0
            else:   # open: keep the sky, and the bounds hold its shell (they still catch splats flung far off)
                c = np.array(bounds["center"])
                reach = 1.5 * scene_scale * 10 + float(np.linalg.norm(centers.mean(0).cpu().numpy() - c))
                bounds["half"] = [max(h, reach) for h in bounds["half"]]
                bounds["open_scene"] = True
                log(f"LiDAR returns over only {bounds['lidar_coverage']:.0%} of the photos: open scene, sky kept")
        if bounds:
            (out / "bounds.json").write_text(json.dumps(bounds, indent=1))
            log("bounds: level box {:.1f} x {:.1f} x {:.1f} m around {} cameras and {:,} LiDAR points".format(
                *(2 * h for h in bounds["half"]), bounds["cameras"], bounds["lidar_points"]))
    elif a.bounds not in ("auto", "off"):
        bounds = json.loads(Path(a.bounds).read_text())
    log(f"{len(tv)} training views, {len(ev)} held out, scene scale {scene_scale:.2f}, "
        f"depth maps {'yes' if tv.has_depth else 'no'}")

    weights = torch.ones(len(tv))
    if a.view_weights:
        vw = json.loads(Path(a.view_weights).read_text())
        weights = torch.tensor([float(vw.get(n, 1.0)) for n in tv.names]).clamp_min(1e-3)
    first = None
    if a.first_names and a.first_until > 0:
        fn = {l.strip() for l in Path(a.first_names).read_text().splitlines() if l.strip()}
        first = torch.tensor([float(n in fn) for n in tv.names])
        if first.sum() == 0:
            first = None
        else:
            log(f"first {a.first_until:.0%} of training on {int(first.sum())} photos only")

    mcmc = a.strategy in ("mcmc", "hybrid")
    if splats is None and a.init_ply:
        splats = gsdata.load_ply(a.init_ply)
    if splats is None:
        if a.points:
            pts, rgb = gsdata.read_points(a.points)
        if len(pts) == 0:
            raise SystemExit("no points3D to initialise from")
        if len(pts) > a.max_splats // 2:
            sel = torch.randperm(len(pts))[: a.max_splats // 2].numpy()
            pts, rgb = pts[sel], rgb[sel]
        if a.sky_points:
            sky, sky_rgb = sky_shell(a.sky_points, centers.mean(0).cpu().numpy(), scene_scale * 10)
            pts, rgb = np.concatenate([pts, sky]), np.concatenate([rgb, sky_rgb])
        splats = init_from_points(pts, rgb, a.sh_degree, 0.5 if mcmc else 0.1, 0.1 if mcmc else 1.0, dev)
        log(f"initialised {len(pts):,} splats from the sparse points")
    params = make_params(splats, dev)
    sh_degree = min(a.sh_degree, gsdata.sh_degree_of(splats))
    fine_tune = a.strategy == "none"

    lrs = {"means": a.means_lr * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    sweeps = None
    if a.sweep_correction:
        sweeps = find_sweeps(tv.names, torch.linalg.inv(tv.viewmats), a.sweep_jump, a.sweep_turn, a.sweep_max)
        log(f"sweep correction: {max(sweeps) + 1} sweeps over {len(tv)} photos")
    pose = (PoseCorrection(len(tv), dev, sweeps, tv.viewmats)
            if a.pose_opt or a.sweep_correction else None)
    pose_opt = (torch.optim.Adam(pose.per_photo(), lr=a.pose_lr, weight_decay=a.pose_reg)
                if a.pose_opt else None)
    sweep_opt = (torch.optim.Adam(pose.per_sweep(), lr=a.sweep_lr, weight_decay=a.pose_reg)
                 if a.sweep_correction else None)
    decay = a.lr_decay_steps or a.steps
    means_lr0 = lrs["means"]

    def means_lr(step):   # exponential decay to lr_final over `decay` steps, then held
        return means_lr0 * a.lr_final ** (min(step, decay) / max(decay, 1))

    if a.strategy == "hybrid":
        from hybrid import HybridStrategy
        strategy = HybridStrategy(power=a.hybrid_power, cap_max=a.max_splats,
                                  refine_stop_iter=a.refine_stop or int(a.steps * 0.85))
        state = strategy.initialize_state()
    elif mcmc:
        strategy = MCMCStrategy(cap_max=a.max_splats, refine_stop_iter=a.refine_stop or int(a.steps * 0.85))
        state = strategy.initialize_state()
    elif not fine_tune:
        strategy = DefaultStrategy(absgrad=a.strategy == "absgrad", refine_stop_iter=a.refine_stop or a.steps // 2,
                                   key_for_gradient="gradient_2dgs" if a.mode == "2dgs" else "means2d")
        state = strategy.initialize_state(scene_scale=scene_scale)
    else:
        strategy = state = None
    if strategy:
        strategy.check_sanity(params, opts)

    packed = a.mode == "3dgs"
    space = Space(a.space, dev) if a.space else None
    box = Box(a.box, dev) if a.box else Box(bounds, dev) if bounds else None
    boxed = 0
    carved = [0, 0]
    eval_view = ev if len(ev) else tv
    t0, last = time.time(), time.time()
    curve = []
    extra = None
    if a.track_cameras:
        xi, _, _ = gsdata.read_model(gsdata.find_model(a.track_cameras))
        extra = gsdata.Views(a.track_cameras, xi, a.max_res, dev)
    for step in range(1, a.steps + 1):
        if first is not None and step <= a.first_until * a.steps:
            i = int(torch.multinomial(first, 1))
        elif a.view_weights:
            i = int(torch.multinomial(weights, 1))
        else:
            i = int(torch.randint(len(tv), (1,)))
        deg = sh_degree if fine_tune else min(sh_degree, (step - 1) // max(a.sh_ramp, 1))
        vmi = pose.viewmat(tv.viewmats[i], i) if pose is not None else None
        rgb_r, depth, alpha, meta, ex = render(params, tv, i, deg, a.mode, a.antialiased,
                                               absgrad=a.strategy in ("absgrad", "hybrid"), packed=packed, viewmat=vmi)
        gt = tv.image(i)
        if strategy:
            strategy.step_pre_backward(params, opts, state, step, meta)
        mask, pw = tv.masks[i], tv.weights[i]
        if mask is None and pw is None:
            loss = (1 - a.ssim_weight) * (rgb_r - gt).abs().mean() + a.ssim_weight * (1 - gsdata.ssim(rgb_r, gt))
        else:
            keep = mask == 255 if mask is not None else torch.ones_like(gt[..., 0], dtype=torch.bool)
            empty = mask == 128 if mask is not None else torch.zeros_like(keep)
            wk = keep.float() * (pw.float() * pw_scale if pw is not None else 1)
            if pw is not None and a.pixel_weights_absolute:   # weights also set the photo's share
                n = keep.float().sum().clamp_min(1.0)
                l1 = ((rgb_r - gt).abs().mean(-1) * wk).sum() / n
                ssim_term = (wk.sum() / n) * (1 - gsdata.ssim(rgb_r, gt, keep, pw))
            else:
                l1 = ((rgb_r - gt).abs().mean(-1) * wk).sum() / wk.sum().clamp_min(1e-6)
                ssim_term = 1 - gsdata.ssim(rgb_r, gt, keep, pw)
            loss = (1 - a.ssim_weight) * l1 + a.ssim_weight * ssim_term
            if empty.any() and a.empty_weight > 0:
                loss = loss + a.empty_weight * alpha[empty].mean()
        if a.depth_weight > 0 and tv.depth[i] is not None:
            loss = loss + a.depth_weight * depth_loss(
                depth, alpha, tv.depth[i], tv.conf[i], a.depth_conf, a.depth_deadband,
                align=a.depth_align_after > 0 and step >= a.depth_align_after * a.steps,
                front_only=a.depth_front_only)
        if a.flat_weight > 0 and a.mode == "3dgs":
            loss = loss + a.flat_weight * torch.exp(params["scales"]).min(dim=-1).values.mean()
        if a.mode == "2dgs":
            if step > a.steps // 4:   # normal consistency once geometry has settled
                n_err = 1 - (ex["normals"] * ex["surf_normals"]).sum(-1)
                loss = loss + 0.05 * n_err.mean()
            if step > a.steps // 10:
                loss = loss + 0.01 * ex["distort"].mean()
        if a.needle_weight > 0:
            loss = loss + a.needle_weight * needle_loss(params, a.needle_max)
        if a.streak_weight > 0:
            loss = loss + a.streak_weight * streak_loss(meta, params["means"], a.streak_max,
                                                        space if a.streak_off_surface else None)
        if mcmc:
            loss = (loss + a.opacity_reg * torch.sigmoid(params["opacities"]).mean()
                    + a.scale_reg * torch.exp(params["scales"]).mean())
        if a.opacity_binary > 0 and step >= a.opacity_binary_start * a.steps:
            o = torch.sigmoid(params["opacities"]).clamp(1e-4, 1 - 1e-4)
            loss = loss - a.opacity_binary * (o * o.log() + (1 - o) * (1 - o).log()).mean()
        if space is not None and a.far_flat_weight > 0:
            loss = loss + a.far_flat_weight * far_flat_loss(params, space)
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if pose_opt is not None:
            pose_opt.step()
            pose_opt.zero_grad(set_to_none=True)
        if sweep_opt is not None:
            if step >= a.sweep_start:
                sweep_opt.step()
            sweep_opt.zero_grad(set_to_none=True)
        for g in opts["means"].param_groups:
            g["lr"] = means_lr(step)
        if mcmc and a.noise_stop > 0 and step == int(a.noise_stop * a.steps):
            strategy.noise_lr = 0.0
            log(f"step {step}: splat jitter stopped")
        if mcmc:
            strategy.step_post_backward(params, opts, state, step, meta, lr=means_lr(step))
            if (strategy.refine_stop_iter <= step < a.relocate_until * a.steps
                    and step % strategy.refine_every == 0):   # growth has stopped; keep reusing dead splats
                strategy._relocate_gs(params, opts, state["binoms"].to(params["means"].device))
        elif strategy:
            strategy.step_post_backward(params, opts, state, step, meta, packed=packed)
            if len(params["means"]) > a.max_splats and strategy.refine_stop_iter > step:
                strategy.refine_stop_iter = step   # cap reached: stop growing
                log(f"splat cap {a.max_splats:,} reached at step {step}; densification stopped")

        if box is not None and step % 100 == 0 and step <= a.steps - 200:
            boxed += keep_in_box(params, box)
        if space is not None and a.carve and step % 100 == 0 and step <= a.steps - 200:
            d, sh = carve(params, space, fade=a.carve_fade, band=a.carve_band)
            carved[0] += d
            carved[1] += sh
        if a.streak_pass_every and step % a.streak_pass_every == 0 and step <= a.steps - 500:
            nf = streak_pass(params, tv, deg, a.mode, a.antialiased, a.streak_pass_share, a.streak_pass_fade,
                             space=space if a.streak_off_surface else None,
                             viewmat=pose.viewmat if pose is not None else None)
            log(f"step {step}: streak pass faded {nf:,} splats")
        if step % 100 == 0 or step == a.steps:
            el = time.time() - t0
            log(f"step {step}/{a.steps} splats {len(params['means']):,} loss {float(loss):.4f} "
                f"{step / el:.1f} it/s" + (f" (+{time.time() - last:.0f}s)" if step % 1000 == 0 else ""))
            last = time.time()
        if a.eval_every and (step % a.eval_every == 0 or step == a.steps):
            with torch.no_grad():
                img, *_ = render(params, eval_view, 0, sh_degree, a.mode, a.antialiased)
            gsdata.save_png(out / f"eval_{step}" / (Path(eval_view.names[0]).stem + ".png"), img)
        if a.test_every and len(ev) and step % a.test_every == 0 and step != a.steps:
            with torch.no_grad():
                m = evaluate(params, ev, sh_degree, a.mode, a.antialiased)
                for i, nm in enumerate(ev.names):
                    if Path(nm).stem in a.track_views or nm in a.track_views:
                        img, *_ = render(params, ev, i, sh_degree, a.mode, a.antialiased)
                        gsdata.save_png(out / "track" / f"{Path(nm).stem}_{step:06d}.png", img.clamp(0, 1))
                if extra is not None:
                    for i, nm in enumerate(extra.names):
                        img, *_ = render(params, extra, i, sh_degree, a.mode, a.antialiased)
                        gsdata.save_png(out / "track" / f"{Path(nm).stem}_{step:06d}.png", img.clamp(0, 1))
            curve.append({"step": step, "psnr": m["psnr"], "ssim": m["ssim"], "depth_err": m.get("depth_err"),
                          "holes": m.get("holes"), "in_front": m.get("in_front"), "streaks": m.get("streaks"),
                          "shapes": shape_stats(params), "splats": len(params["means"]), "seconds": round(time.time() - t0, 1),
                          "views": [{k: v[k] for k in ("name", "psnr", "ssim")} for v in m["views"]]})
            (out / "curve.json").write_text(json.dumps(curve, indent=1))
            log(f"step {step}: held-out PSNR {m['psnr']:.2f} SSIM {m['ssim']:.4f}" + scene_counts(m, params))
        if a.export_every and step % a.export_every == 0 and step != a.steps:
            gsdata.save_ply(out / f"export_{step}.ply", dict(params))

    if a.test_every and (a.track_views or extra is not None):   # final frame of the tracked views
        with torch.no_grad():
            for vw in ([ev] if len(ev) else []) + ([extra] if extra is not None else []):
                for i, nm in enumerate(vw.names):
                    if vw is extra or Path(nm).stem in a.track_views or nm in a.track_views:
                        img, *_ = render(params, vw, i, sh_degree, a.mode, a.antialiased)
                        gsdata.save_png(out / "track" / f"{Path(nm).stem}_{a.steps:06d}.png", img.clamp(0, 1))
    dropped = {"outside": 0, "invisible": 0}
    with torch.no_grad():   # drop what ended up outside the box, and splats too faint to show
        keep = torch.ones(len(params["means"]), dtype=torch.bool, device=dev)
        if box is not None:
            keep = box.inside(params["means"])
            dropped["outside"] = int((~keep).sum())
        if not a.keep_invisible:
            vis = torch.sigmoid(params["opacities"]) >= 1 / 255
            dropped["invisible"] = int((keep & ~vis).sum())
            keep &= vis
        if not keep.all():
            params = make_params({k: v[keep] for k, v in params.items()}, dev)
            log(f"dropped {dropped['outside']:,} splats outside the bounds and {dropped['invisible']:,} invisible ones; "
                f"{len(params['means']):,} left")
    gsdata.save_ply(out / f"export_{a.steps}.ply", dict(params))
    pose_report = None
    if pose is not None:   # how far each training photo's camera moved
        with torch.no_grad():
            sh_mm = (pose.shift.norm(dim=-1) * 1000).cpu().tolist()
            rot_deg = (pose.rot.norm(dim=-1) * 180 / math.pi).cpu().tolist()
        pose_report = [{"name": n, "shift_mm": round(s_, 2), "turn_deg": round(r_, 3)}
                       for n, s_, r_ in zip(tv.names, sh_mm, rot_deg)]
        ss = sorted(sh_mm)
        if a.pose_opt:
            log(f"pose corrections: median shift {ss[len(ss) // 2]:.1f} mm, max {ss[-1]:.1f} mm; "
                f"median turn {sorted(rot_deg)[len(rot_deg) // 2]:.2f} deg, max {max(rot_deg):.2f} deg")
    sweep_report = None
    if pose is not None and pose.sweeps is not None:   # how far each sweep moved as a whole
        with torch.no_grad():
            g_mm = (pose.sweep_shift.norm(dim=-1) * 1000).cpu().tolist()
            g_deg = (pose.sweep_rot.norm(dim=-1) * 180 / math.pi).cpu().tolist()
        sweep_report = []
        for g in range(len(g_mm)):
            members = sorted(n for n, s_ in zip(tv.names, pose.sweeps.tolist()) if s_ == g)
            sweep_report.append({"photos": len(members), "first": members[0], "last": members[-1],
                                 "anchor": g == pose.anchor, "shift_mm": round(g_mm[g], 2),
                                 "turn_deg": round(g_deg[g], 3)})
        log("sweep corrections: " + ", ".join(f"{r['shift_mm']:.0f} mm/{r['turn_deg']:.2f} deg"
                                              for r in sweep_report if not r["anchor"]))
    metrics = {"steps": a.steps, "splats": len(params["means"]), "train_seconds": round(time.time() - t0, 1),
               "pose_corrections": pose_report if a.pose_opt else None,
               "sweep_corrections": sweep_report,
               "peak_gpu_gb": round((torch.cuda.max_memory_allocated() if DEVICE == "cuda"
                                     else torch.mps.driver_allocated_memory()) / 1e9, 2),
               "carved": {"removed": carved[0], "shortened": carved[1]}, "left_box": boxed, "dropped": dropped,
               "bounds": bounds,
               "shapes": shape_stats(params),
               "views_train": len(tv), "views_test": len(ev), "settings": vars(a)}
    if len(ev):
        metrics["test"] = evaluate(params, ev, sh_degree, a.mode, a.antialiased, out / "test_renders")
        log(f"held-out PSNR {metrics['test']['psnr']:.2f} SSIM {metrics['test']['ssim']:.4f}"
            + (f" depth error {100 * metrics['test']['depth_err']:.1f}%" if "depth_err" in metrics["test"] else "")
            + scene_counts(metrics["test"], params))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return params, metrics


if __name__ == "__main__":
    train(parse())
