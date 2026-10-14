"""Post-training pruning for a trained splat (any trainer's 3DGS PLY), on a CUDA machine.

  python prune.py <in.ply> <data> --out <dir> [--score lightgaussian|pup] [--keep 0.5 | --target N]
                  [--finetune 3000] [--depth-dir <sensor_data> --depth-weight 0.1] [--flat-weight 0]
                  [--holdout-every 0] [--eval-res 1440] [--score-res 960]

Scores every splat against the capture's photos, keeps the best, optionally fine-tunes the
survivors (no densification) so they cover for the removed ones, and reports how faithful
each stage is to the photos. Writes <out>/pruned.ply and <out>/prune.json.

Scores
  lightgaussian  global significance (LightGaussian, 2023): each splat's total blending
                 weight over every pixel of every photo, times a mild volume term.
  pup            sensitivity (PUP-3DGS, 2024): log-determinant of the Fisher information of
                 each splat's position and scale (made dimensionless), estimated with random-sign
                 probes of the rendered photos (Hutchinson) rather than the paper's exact per-pixel
                 version. On our captures it keeps less fidelity than lightgaussian, the default.
"""
import argparse
import json
import time
from pathlib import Path

import torch

import gsdata
import train as T


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ply")
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--score", choices=["lightgaussian", "pup"], default="lightgaussian")
    p.add_argument("--keep", type=float, default=0.5, help="fraction of splats to keep")
    p.add_argument("--target", type=int, default=0, help="number of splats to keep (overrides --keep)")
    p.add_argument("--finetune", type=int, default=3000, help="fine-tuning steps after pruning (0 = none)")
    p.add_argument("--depth-dir")
    p.add_argument("--depth-weight", type=float, default=0.0)
    p.add_argument("--depth-conf", type=int, default=2)
    p.add_argument("--flat-weight", type=float, default=0.0)
    p.add_argument("--holdout-every", type=int, default=0)
    p.add_argument("--eval-res", type=int, default=1440)
    p.add_argument("--score-res", type=int, default=960)
    p.add_argument("--probes", type=int, default=8, help="random probes per photo for pup")
    p.add_argument("--volume-power", type=float, default=0.1)
    p.add_argument("--eval-only", action="store_true", help="just report fidelity of <in.ply>")
    p.add_argument("--save-renders", action="store_true")
    return p.parse_args(argv)


def lightgaussian_score(params, views, sh_degree, volume_power):
    n = len(params["means"])
    ones = torch.ones(n, 1, device="cuda", requires_grad=True)
    acc = torch.zeros(n, device="cuda")
    for i in range(len(views)):
        if ones.grad is not None:
            ones.grad = None
        rgb, *_ = T.render(params, views, i, None, colors=ones)
        rgb.sum().backward()
        acc += ones.grad[:, 0]
    if volume_power:
        vol = torch.exp(params["scales"].detach()).prod(-1)
        vol = (vol / torch.quantile(vol[torch.randperm(n, device="cuda")[:100000]], 0.9)).clamp(max=1)
        acc = acc * vol ** volume_power
    return acc


def pup_score(params, views, sh_degree, probes):
    n = len(params["means"])
    means = params["means"].detach().clone().requires_grad_(True)
    scales = torch.exp(params["scales"].detach()).requires_grad_(True)
    H = torch.zeros(n, 6, 6, device="cuda")
    for i in range(len(views)):
        for _ in range(probes):
            rgb, *_ = T.render(params, views, i, sh_degree, means=means, scales=scales)
            r = torch.randint(0, 2, rgb.shape, device="cuda", dtype=rgb.dtype) * 2 - 1
            gm, gs = torch.autograd.grad((rgb * r).sum(), [means, scales])
            # Dimensionless parameters (position in units of the splat's own size, log scale):
            # in metres, the log-determinant falls with size^-12 and big splats always lose.
            s = scales.detach()
            g = torch.cat([gm * s.prod(-1, keepdim=True) ** (1 / 3), gs * s], 1)
            H += g[:, :, None] * g[:, None, :]
    H /= len(views) * probes
    diag = H.diagonal(dim1=1, dim2=2)
    lam = float(diag[diag > 0].median()) * 1e-6 if (diag > 0).any() else 1e-12
    H += lam * torch.eye(6, device="cuda")
    return torch.linalg.slogdet(H).logabsdet


def subset(params, keep):
    return {k: v.detach()[keep] for k, v in params.items()}


def fidelity(params, views, sh_degree, out=None):
    m = T.evaluate(params, views, sh_degree, "3dgs", False, out)
    r = {"psnr": round(m["psnr"], 3), "ssim": round(m["ssim"], 4)}
    if "depth_err" in m:
        r["depth_err"] = round(m["depth_err"], 4)
    return r


def main(a):
    gsdata.keep_awake()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data)
    images, _, _ = gsdata.read_model(gsdata.find_model(root))
    tr_ims, te_ims = gsdata.split(images, a.holdout_every)
    eval_ims = te_ims or tr_ims   # without a holdout, report fidelity to the training photos
    splats = gsdata.load_ply(a.ply)
    deg = gsdata.sh_degree_of(splats)
    params = T.make_params(splats, "cuda")
    n0 = len(params["means"])
    ev = gsdata.Views(root, eval_ims, a.eval_res, depth_dir=a.depth_dir)
    report = {"input": str(a.ply), "splats_in": n0, "eval_views": len(ev),
              "eval_on": "held-out photos" if te_ims else "training photos"}
    report["before"] = fidelity(params, ev, deg, out / "renders_before" if a.save_renders else None)
    print(f"{n0:,} splats; fidelity {report['before']}", flush=True)
    if a.eval_only:
        (out / "prune.json").write_text(json.dumps(report, indent=1))
        return report

    sv = gsdata.Views(root, tr_ims, a.score_res)
    t1 = time.time()
    if a.score == "pup":
        score = pup_score(params, sv, deg, a.probes)
    else:
        score = lightgaussian_score(params, sv, deg, a.volume_power)
    del sv
    k = min(n0, a.target or int(round(n0 * a.keep)))
    keep = torch.topk(score, k).indices.sort().values
    report.update(score=a.score, score_seconds=round(time.time() - t1, 1), splats_out=k)
    pruned = subset(params, keep)
    del params, score
    torch.cuda.empty_cache()
    pp = T.make_params(pruned, "cuda")
    report["pruned"] = fidelity(pp, ev, deg, out / "renders_pruned" if a.save_renders else None)
    print(f"kept {k:,} ({100 * k / n0:.0f}%) by {a.score}; fidelity {report['pruned']}", flush=True)

    if a.finetune:
        del pp
        ta = T.parse([a.data, "--out", str(out / "finetune"), "--steps", str(a.finetune),
                      "--max-res", str(a.eval_res), "--sh-degree", str(deg), "--strategy", "none",
                      "--holdout-every", str(a.holdout_every), "--eval-every", "0",
                      "--means-lr", "1.6e-6", "--depth-weight", str(a.depth_weight),
                      "--depth-conf", str(a.depth_conf), "--flat-weight", str(a.flat_weight)]
                     + (["--depth-dir", a.depth_dir] if a.depth_dir else []))
        pp, _ = T.train(ta, splats=pruned, log=lambda s: print("  " + s, flush=True))
        report["finetuned"] = fidelity(pp, ev, deg, out / "renders_finetuned" if a.save_renders else None)
        report["finetune_steps"] = a.finetune
        print(f"fine-tuned {a.finetune} steps; fidelity {report['finetuned']}", flush=True)
        (out / "finetune" / f"export_{a.finetune}.ply").unlink(missing_ok=True)
    gsdata.save_ply(out / "pruned.ply", dict(pp))
    report["seconds"] = round(time.time() - t0, 1)
    (out / "prune.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(parse())
