"""Sweep correction from COLMAP feature tracks and LiDAR, no training.

  python feature_sweeps.py <tracks model> <poses model> --depth-dir <sensor_data> --out <result.json>
                           [--box CX CY CZ SX SY SZ --frame meta.json --viewer-unflips]
                           [--write-model <dir>]

<tracks model> is a COLMAP text model whose images.txt lists each photo's feature positions
and which 3D point (track) each belongs to - e.g. refine.py's bundle-adjusted model, converted
with `colmap model_converter --output_type TXT`. <poses model> holds the cameras to correct
(same photo names, one PINHOLE camera per photo in the photos' own pixel geometry).

A track is one physical feature (a corner, a printed mark) recognised in many photos. Each
photo that sees it gives one 3D guess of where it is: the ray from that camera through the
feature, out to the depth that photo's own LiDAR measured there. A sweep's guesses move rigidly
with the sweep's cameras, so where two sweeps disagree about the same features, the rigid move
that brings one sweep's guesses onto the other's is the sweep's correction.

  1. lift every feature to 3D with its photo's pose and LiDAR depth (only confident, smooth
     depth - LiDAR smears edges, where features like to sit)
  2. per sweep and track: the median of that sweep's guesses
  3. the largest sweep is the anchor; every other sweep gets the rigid move (Kabsch, with
     outlier tracks dropped and refitted) that best puts its track positions onto the
     consensus of the other sweeps; repeated a few times so the consensus settles

Seconds instead of a training per sweep. Writes each sweep's move (shift at the sweep's centre
and turn) to --out, and with --write-model the corrected poses model.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

import box as boxmod
import gsdata
import train as T


def read_tracks(images_txt):
    """{photo name: [(x, y, track id), ...]} for features that belong to a track."""
    lines = [l for l in open(images_txt) if not l.startswith("#")]
    out, k = {}, 0
    while k < len(lines) - 1:
        head = lines[k].split()
        if len(head) >= 10:
            v = lines[k + 1].split()
            obs = [(float(v[i]), float(v[i + 1]), int(v[i + 2])) for i in range(0, len(v) - 2, 3)]
            out[head[9]] = [o for o in obs if o[2] >= 0]
        k += 2
    return out


def kabsch(A, B, w=None):
    """Rigid R, t minimising sum w |R a + t - b|^2."""
    w = np.ones(len(A)) if w is None else w
    ca, cb = (A * w[:, None]).sum(0) / w.sum(), (B * w[:, None]).sum(0) / w.sum()
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def robust_rigid(A, B, iters=5):
    """Kabsch, refitted without tracks far off (3x the median residual, at least 3 mm)."""
    keep = np.ones(len(A), bool)
    for _ in range(iters):
        R, t = kabsch(A[keep], B[keep])
        r = np.linalg.norm(A @ R.T + t - B, axis=1)
        keep = r <= max(3 * np.median(r[keep]), 0.003)
    return R, t, keep, r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tracks")
    p.add_argument("poses")
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--box", type=float, nargs=6, help="only use features inside this box")
    p.add_argument("--frame")
    p.add_argument("--viewer-unflips", action="store_true")
    p.add_argument("--exclude-names", help="photos to leave out (one per line)")
    p.add_argument("--min-shared", type=int, default=8, help="tracks a sweep must share with the rest")
    p.add_argument("--shift-only", action="store_true",
                   help="fit only a shift per sweep (no turn): steadier when a sweep's shared features "
                        "cover a small area, where a turn is barely pinned down")
    p.add_argument("--write-model")
    a = p.parse_args(argv)

    tracks = read_tracks(Path(a.tracks) / "images.txt")
    images = gsdata.read_model(Path(a.poses))[0]
    drop = {l.strip() for l in open(a.exclude_names)} if a.exclude_names else set()
    stem = lambda n: Path(n).stem
    drop = {stem(n) for n in drop}
    images = [im for im in images if stem(im["name"]) not in drop]
    by_stem = {stem(n): v for n, v in tracks.items()}
    names = [im["name"] for im in images]
    vm = np.stack([im["viewmat"] for im in images])
    sid = np.array(T.find_sweeps(names, torch.linalg.inv(torch.tensor(vm, dtype=torch.float32))))
    box = None
    if a.box:
        meta = json.loads(Path(a.frame).read_text()) if a.frame else None
        box = boxmod.box_in_training(a.box[:3], a.box[3:], 0.0, meta, a.viewer_unflips)

    # 1. every feature lifted to 3D with its own photo's pose and LiDAR
    obs = []   # (track, sweep, photo index, x, y, z)
    for k, im in enumerate(images):
        feats = by_stem.get(stem(im["name"]))
        d, c = gsdata.read_depth(a.depth_dir, im["name"])
        if not feats or d is None:
            continue
        d, c = np.asarray(d), np.asarray(c)
        H, W = d.shape
        cam = im["cam"]
        c2w = np.linalg.inv(im["viewmat"])
        for x, y, tid in feats:
            u, v = int(x * W / cam["w"]), int(y * H / cam["h"])
            if not (1 <= u < W - 1 and 1 <= v < H - 1) or c[v, u] < 2:
                continue
            nb = d[v - 1:v + 2, u - 1:u + 2]
            z = d[v, u]
            if z <= 0.05 or nb.max() - nb.min() > max(0.01, 0.03 * z):   # an edge: LiDAR is smeared there
                continue
            P = c2w[:3, :3] @ np.array([(x - cam["cx"]) / cam["fx"] * z, (y - cam["cy"]) / cam["fy"] * z, z]) + c2w[:3, 3]
            if box is not None and not (np.abs((P - box[0]) @ box[1]) <= box[2]).all():
                continue
            obs.append((tid, sid[k], k, *P))
    obs = np.array(obs)
    print(f"{len(obs):,} feature observations on confident, smooth LiDAR, "
          f"{len(np.unique(obs[:, 0])):,} tracks, {len(np.unique(sid))} sweeps", flush=True)

    # 2. per sweep and track, the median guess
    pos = {}
    for (tid, s), grp in _groups(obs):
        pos.setdefault(int(s), {})[int(tid)] = np.median(grp[:, 3:6], axis=0)
    sweeps = sorted(pos)
    anchor = max(sweeps, key=lambda s: (sid == s).sum())
    moves = {s: (np.eye(3), np.zeros(3)) for s in sweeps}

    # 3. each sweep onto the consensus of the others, a few rounds
    for rnd in range(5):
        moved = {s: {t: moves[s][0] @ X + moves[s][1] for t, X in pos[s].items()} for s in sweeps}
        for s in sweeps:
            if s == anchor:
                continue
            shared, A, B = [], [], []
            for t, X in pos[s].items():
                others = [moved[o][t] for o in sweeps if o != s and t in moved[o]]
                if others:
                    A.append(X)
                    B.append(np.median(others, axis=0))
            if len(A) < a.min_shared:
                continue
            if a.shift_only:
                d = np.array(B) - np.array(A)
                R, t_ = np.eye(3), np.median(d, axis=0)
            else:
                R, t_, keep, r = robust_rigid(np.array(A), np.array(B))
            moves[s] = (R, t_)

    report = {"anchor": int(anchor), "observations": len(obs), "sweeps": {}}
    c2w_all = np.linalg.inv(vm)
    for s in sweeps:
        members = [n for n, q in zip(names, sid) if q == s]
        centre = c2w_all[sid == s][:, :3, 3].mean(0)
        R, t_ = moves[s]
        shift = R @ centre + t_ - centre   # where the sweep's centre goes
        turn = math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        A = np.array(list(pos[s].values()))
        shared = sum(1 for t in pos[s] if any(t in pos[o] for o in sweeps if o != s))
        before = _disagreement(pos, s, sweeps, {o: (np.eye(3), np.zeros(3)) for o in sweeps})
        after = _disagreement(pos, s, sweeps, moves)
        report["sweeps"][str(s)] = {"photos": len(members), "first": min(members), "last": max(members),
                                    "anchor": s == anchor, "shared_tracks": shared,
                                    "shift_mm": [round(1000 * x, 2) for x in shift],
                                    "shift_size_mm": round(1000 * float(np.linalg.norm(shift)), 2),
                                    "turn_deg": round(turn, 3),
                                    "disagreement_mm": [before, after]}
        print(f"sweep {s:2d} ({len(members):2d} photos, {shared:4d} shared tracks): "
              + ("anchor" if s == anchor else
                 f"move {1000 * np.linalg.norm(shift):5.1f} mm / {turn:.2f} deg") +
              f"; median disagreement {before} -> {after} mm", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))

    if a.write_model:
        out = Path(a.write_model)
        out.mkdir(parents=True, exist_ok=True)
        for f in ("cameras.txt", "points3D.txt"):
            if (Path(a.poses) / f).exists():
                shutil.copyfile(Path(a.poses) / f, out / f)
        lines = [l.rstrip("\n") for l in open(Path(a.poses) / "images.txt")]
        where = {n: s for n, s in zip(names, sid)}
        res, k = [], 0
        while k < len(lines):
            f = lines[k].split()
            if lines[k].startswith("#") or len(f) < 10:
                res.append(lines[k])
                k += 1
                continue
            if f[9] in where:
                R, t_ = moves[where[f[9]]]
                w2c = np.eye(4)
                w2c[:3, :3] = _qvec_to_R([float(x) for x in f[1:5]])
                w2c[:3, 3] = [float(x) for x in f[5:8]]
                G = np.eye(4)
                G[:3, :3], G[:3, 3] = R, t_
                nv = np.linalg.inv(G @ np.linalg.inv(w2c))
                f[1:5] = [f"{x:.12f}" for x in boxmod.rotmat_to_qvec(nv[:3, :3])]
                f[5:8] = [f"{x:.12f}" for x in nv[:3, 3]]
            res += [" ".join(f), lines[k + 1] if k + 1 < len(lines) else ""]
            k += 2
        (out / "images.txt").write_text("\n".join(res) + "\n")
        print(f"corrected poses -> {out}")


def _groups(obs):
    order = np.lexsort((obs[:, 1], obs[:, 0]))
    o = obs[order]
    keys = o[:, :2]
    cut = np.flatnonzero((np.diff(keys, axis=0) != 0).any(1)) + 1
    for g in np.split(o, cut):
        yield (g[0, 0], g[0, 1]), g


def _disagreement(pos, s, sweeps, moves):
    """Median distance (mm) between this sweep's track positions and the other sweeps' median."""
    d = []
    R, t = moves[s]
    for tid, X in pos[s].items():
        others = [moves[o][0] @ pos[o][tid] + moves[o][1] for o in sweeps if o != s and tid in pos[o]]
        if others:
            d.append(np.linalg.norm(R @ X + t - np.median(others, axis=0)))
    return round(1000 * float(np.median(d)), 2) if d else None


def _qvec_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


if __name__ == "__main__":
    main()
