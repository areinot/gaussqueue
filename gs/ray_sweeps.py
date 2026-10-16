"""Sweep correction from COLMAP feature tracks alone: no LiDAR, no training.

  python ray_sweeps.py <tracks model> <poses model> --out <result.json>
                       [--exclude-names <file>] [--write-model <dir>]

<tracks model> is a COLMAP text model whose images.txt lists each photo's features and the
track (3D point id) each belongs to; <poses model> holds the cameras to correct (the same photo
names, PINHOLE cameras in the photos' pixel geometry).

A track is one physical feature recognised in several photos; each photo gives a ray from its
camera through the feature. If a sweep's cameras sit a little off from the rest, its rays miss
the other sweeps' rays. Each sweep gets one shift (all its cameras moved together) chosen so
that every track it shares with other sweeps has its rays meet at one point:

    minimise  sum over shared tracks k, rays i:  | (I - d_i d_i^T) (X_k - C_i - [i in sweep] t) |^2

over the shift t and the track points X_k. That is linear: the points are eliminated in closed
form, leaving a 3x3 solve for t. Tracks whose rays still miss by a lot after the fit (a wrong
match, a reflection) are down-weighted and the fit repeated. The largest sweep is the anchor;
every other sweep is fitted against the rest in turn, a few rounds, so they settle together.

Unlike LiDAR-based fits, this uses only the photos and their poses (the poses set the scale),
and unlike COLMAP's own bundle adjustment the few tracks shared between sweeps aren't outvoted
by the thousands within each sweep: each sweep has just the three numbers of its shift.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import box as boxmod
import feature_sweeps as fs
import gsdata
import train as T


def lidar_agrees(rays, by, depth, tol):
    """Whether a track's triangulated point sits where LiDAR saw a surface. rays: [(photo, C, d)].
    Features seen through glass (refraction) or in reflections triangulate behind the surface
    LiDAR measured, or where LiDAR has no confident reading. LiDAR only judges here; it isn't
    used for the fit. True when the median depth gap over confident, smooth readings is under tol."""
    A = sum(np.eye(3) - np.outer(d, d) for _, _, d in rays)
    X = np.linalg.solve(A + 1e-9 * np.eye(3), sum((np.eye(3) - np.outer(d, d)) @ C for _, C, d in rays))
    gaps = []
    for ph, C, d in rays:
        dm, cm = depth.get(ph, (None, None))
        if dm is None:
            continue
        im = by[ph]
        cam = im["cam"]
        Xc = im["viewmat"][:3, :3] @ X + im["viewmat"][:3, 3]
        if Xc[2] <= 0.05:
            continue
        x, y = cam["fx"] * Xc[0] / Xc[2] + cam["cx"], cam["fy"] * Xc[1] / Xc[2] + cam["cy"]
        H, W = dm.shape
        u, v = int(x * W / cam["w"]), int(y * H / cam["h"])
        if not (1 <= u < W - 1 and 1 <= v < H - 1) or cm[v, u] < 2:
            continue
        nb = dm[v - 1:v + 2, u - 1:u + 2]
        if nb.max() - nb.min() > max(0.01, 0.03 * dm[v, u]):   # an edge: LiDAR is smeared there
            continue
        gaps.append(abs(Xc[2] - dm[v, u]))
    return bool(gaps) and float(np.median(gaps)) < tol


def fit_shift(tracks, s, shifts, sid_of, min_tracks=8, rounds=6):
    """Shift for sweep s (others held at their current shifts). tracks: {tid: [(photo, C, d)]}.
    Returns (shift, tracks used, median miss before, median miss after) or None."""
    use = []
    for tid, rays in tracks.items():
        ins = [(C, d) for p, C, d in rays if sid_of[p] == s]
        out = [(C + shifts[sid_of[p]], d) for p, C, d in rays if sid_of[p] != s]
        if ins and out:
            use.append((ins, out))
    if len(use) < min_tracks:
        return None
    w = np.ones(len(use))

    def misses(t):
        r = []
        for ins, out in use:
            rays = [(C + t, d) for C, d in ins] + out
            A = sum(np.eye(3) - np.outer(d, d) for _, d in rays)
            X = np.linalg.solve(A, sum((np.eye(3) - np.outer(d, d)) @ C for C, d in rays))
            r.append(np.median([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - C)) for C, d in rays]))
        return np.array(r)

    before = misses(np.zeros(3))
    t = np.zeros(3)
    for _ in range(rounds):
        M, v = np.zeros((3, 3)), np.zeros(3)
        for wk, (ins, out) in zip(w, use):
            Ps = [np.eye(3) - np.outer(d, d) for _, d in ins]
            Po = [np.eye(3) - np.outer(d, d) for _, d in out]
            A = sum(Ps) + sum(Po)
            if np.linalg.cond(A) > 1e5:
                continue
            B = sum(Ps)
            c = sum(P @ C for P, (C, _) in zip(Ps, ins)) + sum(P @ C for P, (C, _) in zip(Po, out))
            e = sum(P @ C for P, (C, _) in zip(Ps, ins))
            Ai = np.linalg.inv(A)
            M += wk * (B - B @ Ai @ B)
            v += wk * (B @ Ai @ c - e)
        t = np.linalg.solve(M, v)
        r = misses(t)
        scale = max(1.5 * np.median(r), 0.002)   # Huber-like: far-off tracks count less
        w = np.where(r <= scale, 1.0, scale / r)
    return t, len(use), float(np.median(before)), float(np.median(misses(t)))


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def moved(rays, sid_of, moves, centres):
    """Rays [(photo, C, d)] with each sweep's current move (R, t about its centre) applied."""
    out = []
    for p, C, d in rays:
        R, t = moves[sid_of[p]]
        c = centres[sid_of[p]]
        out.append((p, c + R @ (C - c) + t, R @ d))
    return out


def fit_rigid(tracks, s, moves, centres, sid_of, turns=True, min_tracks=8, iters=8, min_turn_info=1e-3):
    """Turn (about the sweep's centre) and shift for sweep s, the others held at their moves.

    Gauss-Newton on the point-to-ray distances |d x (X - C)| of all tracks s shares: for small
    increments, a ray (C, d) of the sweep becomes (C + w x (C - c) + t, d + w x d), which is
    linear in the turn w and shift t; the track points X are eliminated per track (Schur
    complement), leaving a 6x6 solve. Far-off tracks are down-weighted each round. If the
    shared tracks can't pin the turn down (they cover too small an area), only the shift is fitted.
    Returns (R, t, tracks used, median miss before, after, turn fitted) or None."""
    use = [rays for rays in tracks.values()
           if any(sid_of[p] == s for p, _, _ in rays) and any(sid_of[p] != s for p, _, _ in rays)]
    if len(use) < min_tracks:
        return None
    c = centres[s]
    R0, t0 = moves[s]
    R, t = R0.copy(), t0.copy()
    w8 = np.ones(len(use))

    def current(rays):
        trial = dict(moves)
        trial[s] = (R, t)
        return moved(rays, sid_of, trial, centres)

    def triangulate(rays):
        A = sum(np.eye(3) - np.outer(d, d) for _, _, d in rays)
        b = sum((np.eye(3) - np.outer(d, d)) @ C for _, C, d in rays)
        return np.linalg.solve(A + 1e-9 * np.eye(3), b), np.linalg.cond(A)

    def misses():
        r = []
        for rays in use:
            mv = current(rays)
            X, _ = triangulate(mv)
            r.append(np.median([np.linalg.norm(np.cross(d, X - C)) for _, C, d in mv]))
        return np.array(r)

    before = misses()
    do_turn = turns
    for it in range(iters):
        H = np.zeros((6, 6))
        g = np.zeros(6)
        for wk, rays in zip(w8, use):
            mv = current(rays)
            X, cond = triangulate(mv)
            if cond > 1e5:
                continue
            Hxx, Hxp, gx = np.zeros((3, 3)), np.zeros((3, 6)), np.zeros(3)
            Hpp, gp = np.zeros((6, 6)), np.zeros(6)
            for p, C, d in mv:
                r0 = np.cross(d, X - C)
                Jx = skew(d)
                Hxx += Jx.T @ Jx
                gx += Jx.T @ r0
                if sid_of[p] == s:
                    v, wv = X - C, C - c
                    Jw = (np.outer(d, v) - (d @ v) * np.eye(3)) - ((d @ wv) * np.eye(3) - np.outer(wv, d))
                    Jp = np.hstack([Jw, -skew(d)])
                    Hxp += Jx.T @ Jp
                    Hpp += Jp.T @ Jp
                    gp += Jp.T @ r0
            Hi = np.linalg.inv(Hxx + 1e-12 * np.eye(3))
            H += wk * (Hpp - Hxp.T @ Hi @ Hxp)
            g += wk * (gp - Hxp.T @ Hi @ gx)
        if do_turn:
            ev = np.linalg.eigvalsh(H[:3, :3] - H[:3, 3:] @ np.linalg.solve(H[3:, 3:], H[3:, :3]))
            if ev.min() < min_turn_info * ev.max() * 1e-2 or ev.min() <= 0:
                do_turn = False   # the turn isn't pinned down: shift only
        if do_turn:
            step = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
            dw, dt = step[:3], step[3:]
        else:
            dw, dt = np.zeros(3), -np.linalg.solve(H[3:, 3:], g[3:])
        dR = rodrigues(dw)
        R, t = dR @ R, dR @ t + dt   # compose the increment (about the centre) onto the move
        r = misses()
        scale = max(1.5 * np.median(r), 0.002)
        w8 = np.where(r <= scale, 1.0, scale / r)
        if np.linalg.norm(dw) < 1e-6 and np.linalg.norm(dt) < 1e-6:
            break
    return R, t, len(use), float(np.median(before)), float(np.median(misses())), do_turn


def write_moved(src, out, moves, centres, sid_of, anchor):
    """Copy the COLMAP text model src to out with each sweep's cameras moved (R, t about its
    centre). Photos are matched to sweeps by name without extension."""
    out.mkdir(parents=True, exist_ok=True)
    for f in ("cameras.txt", "points3D.txt"):
        if (src / f).exists():
            (out / f).write_bytes((src / f).read_bytes())
    lines = [l.rstrip("\n") for l in open(src / "images.txt")]
    res, k = [], 0
    while k < len(lines):
        f = lines[k].split()
        if lines[k].startswith("#") or len(f) < 10:
            res.append(lines[k])
            k += 1
            continue
        s = sid_of.get(Path(f[9]).stem)
        if s is not None and s != anchor:
            Rm, tm = moves[s]
            Rw = fs._qvec_to_R([float(x) for x in f[1:5]])          # world -> camera
            tv = np.array([float(x) for x in f[5:8]])
            C = -Rw.T @ tv
            C = centres[s] + Rm @ (C - centres[s]) + tm               # centre moved
            Rw = Rw @ Rm.T                                            # and turned
            f[1:5] = [f"{x:.12f}" for x in boxmod.rotmat_to_qvec(Rw)]
            f[5:8] = [f"{x:.12f}" for x in -Rw @ C]
        res += [" ".join(f), lines[k + 1] if k + 1 < len(lines) else ""]
        k += 2
    (out / "images.txt").write_text("\n".join(res) + "\n")
    print(f"corrected poses -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tracks")
    p.add_argument("poses")
    p.add_argument("--out", required=True)
    p.add_argument("--exclude-names")
    p.add_argument("--min-tracks", type=int, default=8)
    p.add_argument("--rounds", type=int, default=4, help="passes over all sweeps")
    p.add_argument("--shift-only", action="store_true", help="fit only a shift per sweep, no turn")
    p.add_argument("--lidar-check", metavar="SENSOR_DATA",
                   help="drop shared tracks that don't sit on the LiDAR surface (glass, reflections); "
                        "depth maps in the <poses> photos' geometry")
    p.add_argument("--lidar-tol", type=float, default=0.02, help="metres (default 0.02)")
    p.add_argument("--write-model", help="write the corrected poses model here")
    p.add_argument("--write-from", help="apply the moves to this model instead of <poses> (photos matched by "
                                        "name without extension, e.g. a box dataset cut from the same capture)")
    a = p.parse_args(argv)

    stem = lambda n: Path(n).stem
    drop = {stem(l.strip()) for l in open(a.exclude_names)} if a.exclude_names else set()
    images = [im for im in gsdata.read_model(Path(a.poses))[0] if stem(im["name"]) not in drop]
    names = [im["name"] for im in images]
    vm = np.stack([im["viewmat"] for im in images])
    sid = np.array(T.find_sweeps(names, torch.linalg.inv(torch.tensor(vm, dtype=torch.float32))))
    sid_of = {stem(n): s for n, s in zip(names, sid)}
    by = {stem(im["name"]): im for im in images}

    tracks = {}
    for n, feats in fs.read_tracks(Path(a.tracks) / "images.txt").items():
        im = by.get(stem(n))
        if im is None:
            continue
        c2w = np.linalg.inv(im["viewmat"])
        cam = im["cam"]
        for x, y, tid in feats:
            d = c2w[:3, :3] @ np.array([(x - cam["cx"]) / cam["fx"], (y - cam["cy"]) / cam["fy"], 1.0])
            tracks.setdefault(tid, []).append((stem(n), c2w[:3, 3], d / np.linalg.norm(d)))
    tracks = {k: v for k, v in tracks.items() if len({sid_of[p] for p, _, _ in v}) > 1}   # shared only
    if a.lidar_check:
        depth = {}
        for ph in by:
            dm, cm = gsdata.read_depth(a.lidar_check, by[ph]["name"])
            depth[ph] = (None, None) if dm is None else (np.asarray(dm), np.asarray(cm))
        n0 = len(tracks)
        tracks = {k: v for k, v in tracks.items() if lidar_agrees(v, by, depth, a.lidar_tol)}
        print(f"LiDAR check: kept {len(tracks):,} of {n0:,} shared tracks")
    sweeps = sorted(set(sid))
    anchor = max(sweeps, key=lambda s: (sid == s).sum())
    c2w_all = np.linalg.inv(vm)
    centres = {s: c2w_all[sid == s][:, :3, 3].mean(0) for s in sweeps}
    moves = {s: (np.eye(3), np.zeros(3)) for s in sweeps}
    info = {}
    for rnd in range(a.rounds):
        for s in sweeps:
            if s == anchor:
                continue
            r = fit_rigid(tracks, s, moves, centres, sid_of, not a.shift_only, a.min_tracks)
            if r is not None:
                R, t, n_used, before, after, turned = r
                moves[s] = (R, t)
                info[s] = (n_used, before, after, turned)
    turn_deg = lambda R: float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

    report = {"anchor": int(anchor), "shared_tracks": len(tracks), "sweeps": {}}
    print(f"{len(tracks):,} tracks shared between sweeps; anchor = sweep {anchor}")
    for s in sweeps:
        members = sorted(n for n, q in zip(names, sid) if q == s)
        n_used, before, after, turned = info.get(s, (0, None, None, False))
        R, t = moves[s]
        report["sweeps"][str(s)] = {"photos": len(members), "first": members[0], "last": members[-1],
                                    "anchor": bool(s == anchor), "tracks": int(n_used),
                                    "shift_mm": [round(1000 * float(x), 2) for x in t],
                                    "shift_size_mm": round(1000 * float(np.linalg.norm(t)), 2),
                                    "turn_deg": round(turn_deg(R), 3), "turn_fitted": bool(turned),
                                    "ray_miss_mm": [None if before is None else round(1000 * before, 2),
                                                    None if after is None else round(1000 * after, 2)]}
        t0, t1 = members[0].split("_")[2], members[-1].split("_")[2]
        print(f"  {t0[:2]}:{t0[2:4]}:{t0[4:]}-{t1[2:4]}:{t1[4:]} ({len(members):2d} photos): " +
              ("anchor" if s == anchor else
               (f"shift {1000 * np.linalg.norm(t):5.1f} mm, turn "
                + (f"{turn_deg(R):.2f} deg" if turned else "(not pinned down)")
                + f" from {n_used:4d} tracks; rays miss by {1000 * before:.1f} -> {1000 * after:.1f} mm"
                if n_used else "too few shared tracks")))
    Path(a.out).write_text(json.dumps(report, indent=1))

    if a.write_model:
        write_moved(Path(a.write_from or a.poses), Path(a.write_model), moves, centres, sid_of, anchor)


if __name__ == "__main__":
    main()
