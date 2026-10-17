"""Sweep correction by search: find the rigid move of one sweep's cameras that best matches a
model trained without that sweep.

  python sweep_search.py <ply> <data> --names <sweep photo list> --out <result.json>
                         [--write-model <dir>] [--max-res 640]

The model (a trained .ply, kept frozen) never saw these photos, so where its render from a
sweep camera disagrees with the photo, the sweep disagrees with the rest of the capture. Each
candidate move is a turn about the sweep's centre (the mean of its camera centres) plus a
shift, applied to all its cameras together, as train.py --sweep-correction does. A candidate
is scored by how well the edges of the renders line up with the edges of the photos (normalised
cross-correlation over each photo's box mask, 255 = inside the box), so colour that the model
gets wrong from a new angle matters less than where the edges are.

Search, coarse to fine: shifts on a grid, then turns on a grid around the best shift, then
gradient steps on all six numbers together with the model frozen (the splats can't move to fit
the sweep, so only the sweep can move). Writes the scores along the way and the result to
--out; --write-model also writes <data>'s model with the sweep's cameras moved.
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import gsdata
import train as T


def edges(img):
    """Gradient magnitude of a [H,W,3] image's grey level."""
    g = img.mean(-1)[None, None]
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device)[None, None] / 8
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(2, 3), padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)[0, 0]


def ncc(a, b, m):
    a, b = a[m], b[m]
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


def move(c2w, centre, rot, shift):
    """A rigid move of world space: turn by rot (axis-angle) about centre, then shift."""
    R = T.axis_angle_to_R(rot[None])[0]
    t = centre - R @ centre + shift
    G = torch.eye(4, device=c2w.device)
    G = torch.cat([torch.cat([R, t[:, None]], 1), G[3:]], 0)
    return G @ c2w


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ply")
    p.add_argument("data")
    p.add_argument("--names", required=True, help="the sweep's photos, one name per line")
    p.add_argument("--out", required=True)
    p.add_argument("--write-model", help="also write <data>'s model with the sweep's cameras moved here")
    p.add_argument("--max-res", type=int, default=640)
    p.add_argument("--shift-range", type=float, default=0.012, help="coarse shift search, +- metres")
    p.add_argument("--shift-step", type=float, default=0.004)
    p.add_argument("--turn-range", type=float, default=1.5, help="coarse turn search, +- degrees")
    p.add_argument("--turn-step", type=float, default=0.5)
    p.add_argument("--refine-steps", type=int, default=200)
    a = p.parse_args(argv)
    dev = T.DEVICE
    t0 = time.time()

    want = {l.strip() for l in open(a.names) if l.strip()}
    images, _, _ = gsdata.read_model(gsdata.find_model(a.data))
    sweep = [im for im in images if im["name"] in want]
    if not sweep:
        raise SystemExit("none of the --names photos are in the model")
    views = gsdata.Views(a.data, sweep, a.max_res, dev)
    splats = gsdata.load_ply(a.ply)
    params = {k: v.to(dev) for k, v in splats.items()}
    sh = gsdata.sh_degree_of(splats)
    c2w = torch.linalg.inv(views.viewmats)
    centre = c2w[:, :3, 3].mean(0)
    targets = []
    for i in range(len(views)):
        m = views.masks[i]
        m = (m == 255) if m is not None else torch.ones(views.rgb[i].shape[:2], dtype=torch.bool, device=dev)
        targets.append((edges(views.image(i)), m))

    def score(rot, shift, grad=False):
        tot = 0.0
        for i in range(len(views)):
            vm = torch.linalg.inv(move(c2w[i], centre, rot, shift))
            with torch.set_grad_enabled(grad):
                rgb, _, alpha, _, _ = T.render(params, views, i, sh, viewmat=vm)
                e, m = targets[i]
                s = ncc(edges(rgb.clamp(0, 1)), e, m & (alpha > 0.5))
            if grad:
                (-s / len(views)).backward()
            tot += float(s)
        return tot / len(views)

    zero = torch.zeros(3, device=dev)
    base = score(zero, zero)
    log = {"photos": len(views), "start_score": round(base, 4)}
    print(f"{len(views)} sweep photos; score with no move {base:.4f}", flush=True)

    # 1. shifts on a grid
    n = int(round(a.shift_range / a.shift_step))
    steps = [k * a.shift_step for k in range(-n, n + 1)]
    best = (base, zero.clone())
    t1 = time.time()
    for d in itertools.product(steps, repeat=3):
        sh_ = torch.tensor(d, device=dev, dtype=torch.float32)
        s = score(zero, sh_)
        if s > best[0]:
            best = (s, sh_)
    shift = best[1]
    log["grid_shift"] = {"score": round(best[0], 4), "shift_mm": [round(1000 * float(x), 1) for x in shift],
                         "candidates": len(steps) ** 3, "seconds": round(time.time() - t1, 1)}
    print(f"shift grid ({len(steps) ** 3} candidates, {time.time() - t1:.0f} s): best {best[0]:.4f} "
          f"at {log['grid_shift']['shift_mm']} mm", flush=True)

    # 2. turns on a grid around the best shift
    n = int(round(a.turn_range / a.turn_step))
    turns = [math.radians(k * a.turn_step) for k in range(-n, n + 1)]
    best_t = (best[0], zero.clone())
    t1 = time.time()
    for r in itertools.product(turns, repeat=3):
        rt = torch.tensor(r, device=dev, dtype=torch.float32)
        s = score(rt, shift)
        if s > best_t[0]:
            best_t = (s, rt)
    rot = best_t[1]
    log["grid_turn"] = {"score": round(best_t[0], 4), "turn_deg": [round(math.degrees(float(x)), 2) for x in rot],
                        "candidates": len(turns) ** 3, "seconds": round(time.time() - t1, 1)}
    print(f"turn grid ({len(turns) ** 3} candidates, {time.time() - t1:.0f} s): best {best_t[0]:.4f} "
          f"at {log['grid_turn']['turn_deg']} deg", flush=True)

    # 3. gradient steps on all six, model frozen
    rot_p = torch.nn.Parameter(rot.clone())
    shift_p = torch.nn.Parameter(shift.clone())
    opt = torch.optim.Adam([{"params": [rot_p], "lr": 2e-4}, {"params": [shift_p], "lr": 2e-4}])
    t1 = time.time()
    for k in range(a.refine_steps):
        opt.zero_grad(set_to_none=True)
        s = score(rot_p, shift_p, grad=True)
        opt.step()
    final = score(rot_p.detach(), shift_p.detach())
    rot, shift = rot_p.detach(), shift_p.detach()
    log["refined"] = {"score": round(final, 4), "seconds": round(time.time() - t1, 1)}
    log["result"] = {"shift_mm": [round(1000 * float(x), 2) for x in shift],
                     "turn_deg": [round(math.degrees(float(x)), 3) for x in rot],
                     "shift_size_mm": round(1000 * float(shift.norm()), 2),
                     "turn_size_deg": round(math.degrees(float(rot.norm())), 3),
                     "centre": [float(x) for x in centre]}
    log["seconds"] = round(time.time() - t0, 1)
    print(f"refined ({a.refine_steps} steps, {time.time() - t1:.0f} s): score {final:.4f}; "
          f"move {log['result']['shift_size_mm']} mm / {log['result']['turn_size_deg']} deg "
          f"(shift {log['result']['shift_mm']} mm, turn {log['result']['turn_deg']} deg); "
          f"total {log['seconds']:.0f} s", flush=True)
    Path(a.out).write_text(json.dumps(log, indent=1))

    if a.write_model:   # the whole model, with this sweep's cameras moved
        src = gsdata.find_model(a.data)
        out = Path(a.write_model)
        out.mkdir(parents=True, exist_ok=True)
        (out / "cameras.txt").write_text((src / "cameras.txt").read_text())
        if (src / "points3D.txt").exists():
            (out / "points3D.txt").write_text((src / "points3D.txt").read_text())
        lines = [l.rstrip("\n") for l in open(src / "images.txt")]
        res, k = [], 0
        while k < len(lines):
            l = lines[k]
            f = l.split()
            if l.startswith("#") or len(f) < 10:
                res.append(l)
                k += 1
                continue
            if f[9] in want:
                q = torch.tensor([float(x) for x in f[1:5]])
                tv = torch.tensor([float(x) for x in f[5:8]])
                vm = torch.eye(4)
                vm[:3, :3] = gsdata.qvec_to_rotmat(q) if hasattr(gsdata, "qvec_to_rotmat") else T_qvec(q)
                vm[:3, 3] = tv
                c2 = move(torch.linalg.inv(vm).to(dev), centre, rot, shift)
                nv = torch.linalg.inv(c2).cpu()
                import box as boxmod
                f[1:5] = [f"{x:.12f}" for x in boxmod.rotmat_to_qvec(nv[:3, :3].numpy())]
                f[5:8] = [f"{float(x):.12f}" for x in nv[:3, 3]]
            res.append(" ".join(f))
            res.append(lines[k + 1] if k + 1 < len(lines) else "")
            k += 2
        (out / "images.txt").write_text("\n".join(res) + "\n")
        print(f"model with the sweep moved -> {out}")


def T_qvec(q):
    w, x, y, z = (float(v) for v in q)
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


if __name__ == "__main__":
    main()
