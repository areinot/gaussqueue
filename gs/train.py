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
smallest axis so splats lie flat on surfaces.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterization_2dgs
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
    p.add_argument("--strategy", choices=["default", "absgrad", "mcmc", "none"], default="mcmc")
    p.add_argument("--max-splats", type=int, default=3_000_000)
    p.add_argument("--depth-dir")
    p.add_argument("--eval-depth-dir", help="depth maps for scoring held-out photos (default: --depth-dir)")
    p.add_argument("--depth-weight", type=float, default=0.0)
    p.add_argument("--depth-conf", type=int, default=2)
    p.add_argument("--flat-weight", type=float, default=0.0)
    p.add_argument("--mode", choices=["3dgs", "2dgs"], default="3dgs")
    p.add_argument("--antialiased", action="store_true")
    p.add_argument("--init-ply")
    p.add_argument("--points", help="points3D.txt to initialise from instead of the model's")
    p.add_argument("--sky-points", type=int, default=0,
                   help="also start with this many splats on a distant sphere (sky, far scenery)")
    p.add_argument("--means-lr", type=float, default=1.6e-4, help="times scene scale")
    p.add_argument("--holdout-every", type=int, default=0)
    p.add_argument("--subsample", type=int, default=1, help="train on every Nth photo")
    p.add_argument("--export-every", type=int, default=0)
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


def make_params(splats, device):
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device).contiguous()) for k, v in splats.items()})


def render(params, views, idx, sh_degree, mode="3dgs", antialiased=False, absgrad=False,
           packed=True, colors=None, scales=None, means=None):
    """Render view(s) idx -> (rgb [H,W,3], depth [H,W], alpha [H,W], meta, extras)."""
    w, h = views.size(idx)
    means = params["means"] if means is None else means
    scales = torch.exp(params["scales"]) if scales is None else scales
    opac = torch.sigmoid(params["opacities"])
    sh = colors is None
    if sh:
        colors = torch.cat([params["sh0"], params["shN"]], 1)
    vm, K = views.viewmats[idx:idx + 1], views.K[idx:idx + 1]
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


def depth_loss(depth, alpha, gt, conf, min_conf):
    """Relative L1 between rendered depth (pooled to the LiDAR map's size) and LiDAR depth,
    only where the LiDAR is confident."""
    h, w = gt.shape
    d = F.adaptive_avg_pool2d(depth[None, None], (h, w))[0, 0]
    a = F.adaptive_avg_pool2d(alpha[None, None], (h, w))[0, 0]
    m = (conf >= min_conf) & (gt > 0.05) & torch.isfinite(gt) & (a > 0.5)
    if not m.any():
        return depth.new_zeros(())
    return ((d[m] - gt[m]).abs() / gt[m]).mean()


# ---------------------------------------------------------------- training

def depth_error(depth, alpha, gt, conf):
    """Median relative error of rendered depth against confident LiDAR depth."""
    h, w = gt.shape
    d = F.adaptive_avg_pool2d(depth[None, None], (h, w))[0, 0]
    m = (conf >= 2) & (gt > 0.05) & torch.isfinite(gt)
    if not m.any():
        return None
    a = F.adaptive_avg_pool2d(alpha[None, None], (h, w))[0, 0]
    err = torch.where(a[m] > 0.5, (d[m] - gt[m]).abs() / gt[m], torch.ones_like(gt[m]))
    return float(err.median())


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
            rgb, depth, alpha, *_ = render(params, views, i, sh_degree, mode, antialiased)
            gt = views.image(i)
            rgb = rgb.clamp(0, 1)
            r = {"name": views.names[i], "psnr": gsdata.psnr(rgb, gt), "ssim": float(gsdata.ssim(rgb, gt))}
            if views.depth[i] is not None:
                r["depth_err"] = depth_error(depth, alpha, views.depth[i], views.conf[i])
            res.append(r)
            if out_dir:
                stem = Path(views.names[i]).stem
                gsdata.save_png(Path(out_dir) / f"{stem}.png", rgb)
                gsdata.save_png(Path(out_dir) / f"{stem}_depth.png", depth_image(depth, alpha))
    if not res:
        return {}
    out = {"psnr": sum(r["psnr"] for r in res) / len(res), "ssim": sum(r["ssim"] for r in res) / len(res),
           "views": res}
    de = [r["depth_err"] for r in res if r.get("depth_err") is not None]
    if de:
        out["depth_err"] = sum(de) / len(de)
    return out


def train(a, splats=None, log=print):
    """Train (or fine-tune, when `splats` or --init-ply is given). Returns (params, metrics)."""
    gsdata.keep_awake()
    torch.manual_seed(a.seed)
    dev = "cuda"
    root = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images, pts, rgb = gsdata.read_model(gsdata.find_model(root))
    tr_ims, te_ims = gsdata.split(images, a.holdout_every)
    tr_ims = tr_ims[::max(1, a.subsample)]
    tv = gsdata.Views(root, tr_ims, a.max_res, dev, a.depth_dir if a.depth_weight > 0 else None)
    ev = gsdata.Views(root, te_ims, a.max_res, dev, a.eval_depth_dir or a.depth_dir)
    centers = tv.centers()
    scene_scale = float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1 or 1.0
    log(f"{len(tv)} training views, {len(ev)} held out, scene scale {scene_scale:.2f}, "
        f"depth maps {'yes' if tv.has_depth else 'no'}")

    mcmc = a.strategy == "mcmc"
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
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1 / max(a.steps, 1)))

    if mcmc:
        strategy = MCMCStrategy(cap_max=a.max_splats, refine_stop_iter=int(a.steps * 0.85))
        state = strategy.initialize_state()
    elif not fine_tune:
        strategy = DefaultStrategy(absgrad=a.strategy == "absgrad", refine_stop_iter=a.steps // 2,
                                   key_for_gradient="gradient_2dgs" if a.mode == "2dgs" else "means2d")
        state = strategy.initialize_state(scene_scale=scene_scale)
    else:
        strategy = state = None
    if strategy:
        strategy.check_sanity(params, opts)

    packed = a.mode == "3dgs"
    eval_view = ev if len(ev) else tv
    t0, last = time.time(), time.time()
    for step in range(1, a.steps + 1):
        i = int(torch.randint(len(tv), (1,)))
        deg = sh_degree if fine_tune else min(sh_degree, (step - 1) // 1000)
        rgb_r, depth, alpha, meta, ex = render(params, tv, i, deg, a.mode, a.antialiased,
                                               absgrad=a.strategy == "absgrad", packed=packed)
        gt = tv.image(i)
        if strategy:
            strategy.step_pre_backward(params, opts, state, step, meta)
        loss = 0.8 * (rgb_r - gt).abs().mean() + 0.2 * (1 - gsdata.ssim(rgb_r, gt))
        if a.depth_weight > 0 and tv.depth[i] is not None:
            loss = loss + a.depth_weight * depth_loss(depth, alpha, tv.depth[i], tv.conf[i], a.depth_conf)
        if a.flat_weight > 0 and a.mode == "3dgs":
            loss = loss + a.flat_weight * torch.exp(params["scales"]).min(dim=-1).values.mean()
        if a.mode == "2dgs":
            if step > a.steps // 4:   # normal consistency once geometry has settled
                n_err = 1 - (ex["normals"] * ex["surf_normals"]).sum(-1)
                loss = loss + 0.05 * n_err.mean()
            if step > a.steps // 10:
                loss = loss + 0.01 * ex["distort"].mean()
        if mcmc:
            loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        if mcmc:
            strategy.step_post_backward(params, opts, state, step, meta, lr=sched.get_last_lr()[0])
        elif strategy:
            strategy.step_post_backward(params, opts, state, step, meta, packed=packed)
            if len(params["means"]) > a.max_splats and strategy.refine_stop_iter > step:
                strategy.refine_stop_iter = step   # cap reached: stop growing
                log(f"splat cap {a.max_splats:,} reached at step {step}; densification stopped")

        if step % 100 == 0 or step == a.steps:
            el = time.time() - t0
            log(f"step {step}/{a.steps} splats {len(params['means']):,} loss {float(loss):.4f} "
                f"{step / el:.1f} it/s" + (f" (+{time.time() - last:.0f}s)" if step % 1000 == 0 else ""))
            last = time.time()
        if a.eval_every and (step % a.eval_every == 0 or step == a.steps):
            with torch.no_grad():
                img, *_ = render(params, eval_view, 0, sh_degree, a.mode, a.antialiased)
            gsdata.save_png(out / f"eval_{step}" / (Path(eval_view.names[0]).stem + ".png"), img)
        if a.export_every and step % a.export_every == 0 and step != a.steps:
            gsdata.save_ply(out / f"export_{step}.ply", dict(params))

    gsdata.save_ply(out / f"export_{a.steps}.ply", dict(params))
    metrics = {"steps": a.steps, "splats": len(params["means"]), "train_seconds": round(time.time() - t0, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "views_train": len(tv), "views_test": len(ev), "settings": vars(a)}
    if len(ev):
        metrics["test"] = evaluate(params, ev, sh_degree, a.mode, a.antialiased, out / "test_renders")
        log(f"held-out PSNR {metrics['test']['psnr']:.2f} SSIM {metrics['test']['ssim']:.4f}"
            + (f" depth error {100 * metrics['test']['depth_err']:.1f}%" if "depth_err" in metrics["test"] else ""))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return params, metrics


if __name__ == "__main__":
    train(parse())
