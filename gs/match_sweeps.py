"""Sweep correction from COLMAP's raw feature matches: no LiDAR, no training.

  python match_sweeps.py <database.db> <poses model> --out <result.json>
                         [--exclude-names <file>] [--box <box.json>] [--write-model <dir> [--write-from <model>]]

<database.db> is the COLMAP database of a refine run (features + matches, geometrically
verified); <poses model> the cameras the photos were matched in (same photo names, PINHOLE).

Why not COLMAP's tracks: COLMAP links matches into tracks with the cameras held where they
are, and drops every match that doesn't fit them. A sweep that drifted keeps only the few
matches that happen to agree with its drift, so tracks always say the sweeps agree. The raw
matches don't have that bias, but many are wrong (repeated objects: one cup's logo matched to
another cup's). So instead of a least-squares fit, each sweep's move is found by consensus:

  score(move) = sum over matches between the sweep and the rest of exp(-e^2 / 2 tol^2)

where e is how far (pixels, both photos) the match's two rays miss when triangulated with the
sweep's cameras moved. Wrong matches agree with no single move and only add a flat floor; the
right move lines up every true match at once. The move (turn about the sweep's centre, then a
shift) is searched on grids: shift, then turn, then both again finer. The largest sweep is the
anchor; the others are searched largest first, a couple of rounds so they settle together.
"""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch

import gsdata
import ray_sweeps as rs
import train as T

PAIR = 2147483647


def grid(r, step):
    v = np.arange(-r, r + step / 2, step)
    return np.stack(np.meshgrid(v, v, v, indexing="ij"), -1).reshape(-1, 3)


class Matches:
    """Cross-sweep matches as arrays: for side 0/1, photo index and camera-frame ray."""

    def __init__(self, db, photos, sid, max_matches, raw=False):
        """db: one COLMAP database, or a list (their matches are pooled per photo pair, each
        distinct keypoint pair once; the runs must share keypoints, e.g. repeated runs of the same
        settings). raw: LightGlue's matches before COLMAP's geometric verification. That
        verification is RANSAC, so whole photo pairs pass in one run and fail in the next; the
        raw matches are the same every run, and the searches here reject outliers themselves."""
        dbs = [db] if isinstance(db, (str, Path)) else list(db)
        idx = {Path(n).stem: k for k, n in enumerate(photos["name"])}
        table = "matches" if raw else "two_view_geometries"
        pooled = {}
        kps = {}
        for one in dbs:
            con = sqlite3.connect(one)
            iname = {i: Path(n).stem for i, n in con.execute("select image_id, name from images")}
            for i, r, c, d in con.execute("select image_id, rows, cols, data from keypoints"):
                kps.setdefault(iname[i], np.frombuffer(d, np.float32).reshape(r, c)[:, :2] if d is not None and r
                               else np.zeros((0, 2), np.float32))   # a photo with no keypoints stores no data
            for pid, r, _, data in con.execute(f"select pair_id, rows, cols, data from {table} where rows > 0"):
                i2 = pid % PAIR
                i1 = (pid - i2) // PAIR
                key = (iname[i1], iname[i2])
                a, b = idx.get(key[0]), idx.get(key[1])
                if a is None or b is None or sid[a] == sid[b]:
                    continue
                m = np.frombuffer(data, np.uint32).reshape(r, 2)
                pooled[key] = np.unique(np.concatenate([pooled[key], m]), axis=0) if key in pooled else m
        pa, pb, xa, xb = [], [], [], []
        for (n1, n2), m in pooled.items():
            a, b = idx[n1], idx[n2]
            pa.append(np.full(len(m), a)); pb.append(np.full(len(m), b))
            xa.append(kps[n1][m[:, 0]]); xb.append(kps[n2][m[:, 1]])
        self.p = [np.concatenate(pa), np.concatenate(pb)]
        x = [np.concatenate(xa), np.concatenate(xb)]
        K = photos["K"]
        self.dc = []
        for side in (0, 1):
            k = K[self.p[side]]
            d = np.stack([(x[side][:, 0] - k[:, 2]) / k[:, 0], (x[side][:, 1] - k[:, 3]) / k[:, 1],
                          np.ones(len(k))], -1)
            self.dc.append(d / np.linalg.norm(d, axis=1, keepdims=True))
        self.x = x
        self.max_matches = max_matches


def world(photos, moves, centres, sid, p):
    """Moved camera-to-world rotation and centre of photos p."""
    R0, C0 = photos["R"][p], photos["C"][p]
    Rm = np.stack([moves[s][0] for s in sid[p]])
    tm = np.stack([moves[s][1] for s in sid[p]])
    c = np.stack([centres[s] for s in sid[p]])
    return Rm @ R0, c + np.einsum("nij,nj->ni", Rm, C0 - c) + tm


def miss_px(Ra, Ca, da, Rb, Cb, db, xa, xb, Ka, Kb):
    """Two-ray triangulation (midpoint) and the reprojection miss in both photos, pixels.
    Inputs broadcast over a leading candidate axis. Near-parallel or behind-camera -> inf."""
    w0 = Ca - Cb
    b = (da * db).sum(-1)
    d, e = (da * w0).sum(-1), (db * w0).sum(-1)
    den = 1 - b * b
    s1 = (b * e - d) / np.maximum(den, 1e-9)
    s2 = (e - b * d) / np.maximum(den, 1e-9)
    X = 0.5 * (Ca + s1[..., None] * da + Cb + s2[..., None] * db)
    out = np.zeros(X.shape[:-1])
    for R, C, x, K in ((Ra, Ca, xa, Ka), (Rb, Cb, xb, Kb)):
        Xc = np.einsum("...ji,...j->...i", R, X - C)   # R^T (X - C)
        z = np.maximum(Xc[..., 2], 1e-6)
        u = K[..., 0] * Xc[..., 0] / z + K[..., 2]
        v = K[..., 1] * Xc[..., 1] / z + K[..., 3]
        out += np.hypot(u - x[..., 0], v - x[..., 1])
    bad = (den < 1e-4) | (s1 < 0.03) | (s2 < 0.03)
    return np.where(bad, np.inf, out)


def search_sweep(M, photos, moves, centres, sid, s, a, keep=None):
    """Best (R, t) for sweep s by consensus over its cross-sweep matches."""
    sel = (sid[M.p[0]] == s) | (sid[M.p[1]] == s)
    if keep is not None:
        sel &= keep
    ids = np.flatnonzero(sel)
    if len(ids) > M.max_matches:
        ids = np.random.default_rng(0).choice(ids, M.max_matches, replace=False)
    if len(ids) < 20:
        return None
    # put the sweep's side first
    flip = sid[M.p[0][ids]] != s
    pm = np.where(flip, M.p[1][ids], M.p[0][ids])
    pf = np.where(flip, M.p[0][ids], M.p[1][ids])
    dm = np.where(flip[:, None], M.dc[1][ids], M.dc[0][ids])
    df = np.where(flip[:, None], M.dc[0][ids], M.dc[1][ids])
    xm = np.where(flip[:, None], M.x[1][ids], M.x[0][ids])
    xf = np.where(flip[:, None], M.x[0][ids], M.x[1][ids])
    Km, Kf = photos["K"][pm], photos["K"][pf]
    Rf, Cf = world(photos, moves, centres, sid, pf)
    dfw = np.einsum("nij,nj->ni", Rf, df)
    R0, C0 = photos["R"][pm], photos["C"][pm]
    c = centres[s]

    def scores(Rs, ts, tol, chunk=48):
        out = []
        for k in range(0, len(Rs), chunk):
            R, t = Rs[k:k + chunk, None], ts[k:k + chunk, None]          # (K, 1, 3, 3), (K, 1, 3)
            Rw = R @ R0[None]                                              # (K, N, 3, 3)
            Cw = c + np.einsum("knij,nj->kni", np.broadcast_to(R, (len(R), len(pm), 3, 3)), C0 - c) + t
            dw = np.einsum("knij,nj->kni", Rw, dm)
            e = miss_px(Rw, Cw, dw, Rf[None], Cf[None], dfw[None], xm[None], xf[None], Km[None], Kf[None])
            out.append(np.exp(-0.5 * (e / tol) ** 2).sum(1))
        return np.concatenate(out)

    R, t = moves[s]
    base = scores(R[None], t[None], a.tol)[0]

    def turns(w):
        return np.stack([rs.rodrigues(np.radians(v)) for v in w])

    # coarse to fine; the coarse grids can't land within a few pixels of the peak, so they
    # score with a wider tolerance (the peak is as wide as the grid step)
    ss, ts = a.shift_step, a.turn_step
    for sr, ss_, tr, ts_, tol in ((a.shift_range, ss, a.turn_range, ts, 4 * a.tol),
                                  (ss, ss / 3, ts, ts / 3, 2 * a.tol),
                                  (ss / 3, ss / 9, ts / 3, ts / 9, a.tol)):
        cand = t + grid(sr, ss_)
        sc = scores(np.broadcast_to(R, (len(cand), 3, 3)), cand, tol)
        t = cand[sc.argmax()]
        Rc = turns(grid(tr, ts_)) @ R
        sc = scores(Rc, np.broadcast_to(t, (len(Rc), 3)), tol)
        R = Rc[sc.argmax()]
    best = scores(R[None], t[None], a.tol)[0]
    Rw0, Cw0 = world(photos, moves, centres, sid, pm)
    e0 = miss_px(Rw0, Cw0, np.einsum("nij,nj->ni", Rw0, dm), Rf, Cf, dfw, xm, xf, Km, Kf)
    trial = dict(moves); trial[s] = (R, t)
    Rw, Cw = world(photos, trial, centres, sid, pm)
    e1 = miss_px(Rw, Cw, np.einsum("nij,nj->ni", Rw, dm), Rf, Cf, dfw, xm, xf, Km, Kf)
    return R, t, len(ids), base, best, float((e0 < 2 * a.tol).mean()), float((e1 < 2 * a.tol).mean())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("database")
    p.add_argument("poses")
    p.add_argument("--out", required=True)
    p.add_argument("--exclude-names")
    p.add_argument("--box", help="box.json: use only matches that land in the box (with --box-margin)")
    p.add_argument("--box-margin", type=float, default=0.02)
    p.add_argument("--tol", type=float, default=3.0, help="pixels a true match may miss by (default 3)")
    p.add_argument("--shift-range", type=float, default=0.016, help="metres (default 0.016)")
    p.add_argument("--shift-step", type=float, default=0.002)
    p.add_argument("--turn-range", type=float, default=2.0, help="degrees (default 2)")
    p.add_argument("--turn-step", type=float, default=0.5)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--max-matches", type=int, default=30000, help="per sweep, sampled")
    p.add_argument("--write-model")
    p.add_argument("--write-from", help="apply the moves to this model instead (photos matched by name stem)")
    a = p.parse_args(argv)
    t_start = time.time()

    stem = lambda n: Path(n).stem
    drop = {stem(l.strip()) for l in open(a.exclude_names)} if a.exclude_names else set()
    images = [im for im in gsdata.read_model(Path(a.poses))[0] if stem(im["name"]) not in drop]
    vm = np.stack([im["viewmat"] for im in images])
    c2w = np.linalg.inv(vm)
    names = [im["name"] for im in images]
    sid = np.array(T.find_sweeps(names, torch.tensor(c2w, dtype=torch.float32)))
    photos = {"name": names, "R": c2w[:, :3, :3], "C": c2w[:, :3, 3],
              "K": np.array([[im["cam"][k] for k in ("fx", "fy", "cx", "cy")] for im in images])}
    M = Matches(a.database, photos, sid, a.max_matches)
    sweeps = sorted(set(sid))
    anchor = max(sweeps, key=lambda s: (sid == s).sum())
    centres = {s: photos["C"][sid == s].mean(0) for s in sweeps}
    moves = {s: (np.eye(3), np.zeros(3)) for s in sweeps}
    keep = None
    if a.box:
        b = json.loads(Path(a.box).read_text())
        bc, ax, half = np.array(b["center"]), np.array(b["axes"]), np.array(b["half"]) + a.box_margin
        Ra, Ca = world(photos, moves, centres, sid, M.p[0])
        Rb, Cb = world(photos, moves, centres, sid, M.p[1])
        da, db = np.einsum("nij,nj->ni", Ra, M.dc[0]), np.einsum("nij,nj->ni", Rb, M.dc[1])
        w0 = Ca - Cb
        bb = (da * db).sum(-1)
        s1 = (bb * (db * w0).sum(-1) - (da * w0).sum(-1)) / np.maximum(1 - bb * bb, 1e-9)
        X = Ca + s1[:, None] * da
        keep = (np.abs((X - bc) @ ax.T) <= half).all(1)
    print(f"{len(M.p[0]):,} verified matches between sweeps" +
          (f", {int(keep.sum()):,} in the box" if keep is not None else "") + f"; anchor = sweep {anchor}")

    order = sorted((s for s in sweeps if s != anchor), key=lambda s: -(sid == s).sum())
    info = {}
    for rnd in range(a.rounds):
        for s in order:
            r = search_sweep(M, photos, moves, centres, sid, s, a, keep)
            if r is not None:
                moves[s] = r[:2]
                info[s] = r[2:]
    turn_deg = lambda R: float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

    report = {"anchor": int(anchor), "seconds": round(time.time() - t_start, 1), "sweeps": {}}
    for s in sweeps:
        members = sorted(n for n, q in zip(names, sid) if q == s)
        R, t = moves[s]
        n, base, best, f0, f1 = info.get(s, (0, 0, 0, 0, 0))
        report["sweeps"][str(s)] = {
            "photos": len(members), "first": members[0], "last": members[-1], "anchor": bool(s == anchor),
            "matches": int(n), "shift_mm": [round(1000 * float(x), 2) for x in t],
            "shift_size_mm": round(1000 * float(np.linalg.norm(t)), 2), "turn_deg": round(turn_deg(R), 3),
            "R": R.round(8).tolist(), "centre": centres[s].tolist(),
            "agreeing": [round(f0, 3), round(f1, 3)]}
        t0, t1 = members[0].split("_")[2], members[-1].split("_")[2]
        print(f"  {t0[:2]}:{t0[2:4]}:{t0[4:]}-{t1[2:4]}:{t1[4:]} ({len(members):2d} photos): " +
              ("anchor" if s == anchor else
               f"shift {1000 * np.linalg.norm(t):5.1f} mm, turn {turn_deg(R):.2f} deg from {n:6,d} matches; "
               f"agreeing {100 * f0:.0f}% -> {100 * f1:.0f}%" if n else "too few matches"))
    print(f"{time.time() - t_start:.0f} s")
    Path(a.out).write_text(json.dumps(report, indent=1))
    if a.write_model:
        rs.write_moved(Path(a.write_from or a.poses), Path(a.write_model), moves, centres,
                       {stem(n): s for n, s in zip(names, sid)}, anchor)


if __name__ == "__main__":
    main()
