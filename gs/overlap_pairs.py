"""Photo pairs worth matching, from the poses instead of all pairs (N^2).

  python overlap_pairs.py --model <sparse/0> [--points <points3D.txt>] --out pairs.txt
                          [--own 8] [--cross 30] [--per-sweep 4] [--min-overlap 0.05]

Each photo "sees" the scene points (the model's points3D, e.g. merged LiDAR and feature points,
on a 5 cm voxel grid) that fall inside its view, nearer than --max-depth. Two photos overlap by the
share of the smaller one's voxels they both see, and must look within --max-angle of each other.
Each photo keeps its --own best partners from its own sweep and its --cross best from other sweeps,
at most --per-sweep from any one: links between sweeps are what keep a sweep from drifting, and
within a sweep a few neighbours are enough. Pairs are written as COLMAP pair lines ("a.jpg b.jpg").
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import gsdata
import train as T


def read_points(path):
    pts = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        pts.append((float(f[1]), float(f[2]), float(f[3])))
    return np.array(pts, float).reshape(-1, 3)


def footprints(images, pts, voxel, max_depth):
    """[photos, voxels] bool: which voxels of scene points fall inside each photo's view."""
    keys = np.floor(pts / voxel).astype(np.int64)
    keys, inv = np.unique(keys, axis=0, return_inverse=True)
    centres = (keys + 0.5) * voxel
    out = np.zeros((len(images), len(centres)), bool)
    for k, im in enumerate(images):
        vm = im["viewmat"]
        pc = centres @ vm[:3, :3].T + vm[:3, 3]
        z = pc[:, 2]
        c = im["cam"]
        u = c["fx"] * pc[:, 0] / np.maximum(z, 1e-6) + c["cx"]
        v = c["fy"] * pc[:, 1] / np.maximum(z, 1e-6) + c["cy"]
        ok = (z > 0.05) & (z < max_depth) & (u >= 0) & (u < c["w"]) & (v >= 0) & (v < c["h"])
        out[k] = ok
    return out


def select(names, sweep, fwd, foot, own, cross, per_sweep, min_overlap, max_angle):
    N = len(names)
    cosmax = np.cos(np.radians(max_angle))
    F = torch.from_numpy(foot.astype(np.float32))
    shared = (F @ F.T).numpy()                                  # voxels both photos see
    sizes = foot.sum(1)
    ov_all = shared / np.maximum(np.minimum(sizes[:, None], sizes[None, :]), 1)
    ov_all[fwd @ fwd.T < cosmax] = 0
    np.fill_diagonal(ov_all, 0)
    pairs = set()
    stats = defaultdict(int)
    for i in range(N):
        if not sizes[i]:
            continue
        js = np.flatnonzero(ov_all[i] >= min_overlap)
        cand = sorted(zip(ov_all[i, js], js), reverse=True)
        n_own, n_cross, per = 0, 0, defaultdict(int)
        for ov, j in cand:
            if sweep[j] == sweep[i]:
                if n_own >= own:
                    continue
                n_own += 1
            else:
                if n_cross >= cross or per[sweep[j]] >= per_sweep:
                    continue
                n_cross += 1
                per[sweep[j]] += 1
            pairs.add((min(i, j), max(i, j)))
        stats["own"] += n_own
        stats["cross"] += n_cross
    return sorted(pairs), stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="COLMAP text model with the poses")
    p.add_argument("--points", help="points3D.txt for the scene (default: the model's own)")
    p.add_argument("--out", required=True)
    p.add_argument("--own", type=int, default=8)
    p.add_argument("--cross", type=int, default=30)
    p.add_argument("--per-sweep", type=int, default=4)
    p.add_argument("--min-overlap", type=float, default=0.05)
    p.add_argument("--max-angle", type=float, default=75.0, help="deg between viewing directions")
    p.add_argument("--max-depth", type=float, default=3.0, help="m")
    p.add_argument("--voxel", type=float, default=0.05, help="m")
    p.add_argument("--sweep-jump", type=float, default=0.0)
    p.add_argument("--sweep-min", type=int, default=3)
    a = p.parse_args(argv)
    images = gsdata.read_model(Path(a.model))[0]
    images.sort(key=lambda im: Path(im["name"]).stem)
    names = [im["name"] for im in images]
    pts = read_points(a.points or Path(a.model) / "points3D.txt")
    if not len(pts):
        raise SystemExit("no scene points: pass --points with a points3D.txt that has some")
    c2w = np.linalg.inv(np.stack([im["viewmat"] for im in images]))
    sweep = np.array(T.find_sweeps(names, torch.tensor(c2w, dtype=torch.float32), a.sweep_jump, min_size=a.sweep_min))
    fwd = c2w[:, :3, 2]
    foot = footprints(images, pts, a.voxel, a.max_depth)
    pairs, st = select(names, sweep, fwd, foot, a.own, a.cross, a.per_sweep, a.min_overlap, a.max_angle)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(f"{names[i]} {names[j]}\n" for i, j in pairs))
    N = len(names)
    print(f"{N} photos, {len(set(sweep))} sweeps: {len(pairs):,} pairs of {N * (N - 1) // 2:,} "
          f"({100 * len(pairs) / max(N * (N - 1) // 2, 1):.1f}%); per photo {st['own'] / N:.1f} own-sweep, "
          f"{st['cross'] / N:.1f} cross-sweep picks", flush=True)


if __name__ == "__main__":
    main()
