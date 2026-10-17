"""Sweep correction from COLMAP's raw matches, with the turn and the shift found separately.

  python sep_sweeps.py <database.db> <poses model> --out <result.json>
                       [--exclude-names <file>] [--box <box.json>] [--write-model <dir> [--write-from <model>]]
                       [--sweep-jump M --sweep-max N --sweep-min N]

Same input and output as match_sweeps.py, different solver. A sweep's move is a turn about its
centre plus a shift. On screen a shift moves a point by about f * shift / depth while a turn
moves every point by about f * angle, so far matches measure the turn almost free of the shift
and near ones, once the turn is known, measure the shift. For each sweep:

  1. triangulate its cross-sweep matches with the current cameras; split them by depth
     (farthest third, nearest third)
  2. turn: RANSAC over the far matches (three matches -> one Gauss-Newton step for the turn,
     scored by how many far matches then reproject within the tolerance), refined on the inliers
  3. shift: the same over the near matches, with the turn held
  4. 2-3 rounds of 2+3 with the tolerance tightening (the first turn still carries some of the
     shift on a shallow scene), then a joint robust fit of all six on every inlier

Wrong matches (a cup's logo matched to another cup's) agree with no hypothesis and drop out in
the RANSAC; far ones can only vote on the turn and near ones only on the shift. Reported per
sweep: the depth ratio far/near (how well the two separate: ~1 means one depth and the turn is
weakly pinned), inliers, and the move. Runs on the GPU (CUDA or Apple MPS) or CPU via PyTorch.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import gsdata
import match_sweeps as ms
import ray_sweeps as rs
import train as T

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def skew(v):   # [..., 3] -> [..., 3, 3]
    z = torch.zeros_like(v[..., 0])
    return torch.stack([z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0], -v[..., 1], v[..., 0], z], -1).view(*v.shape[:-1], 3, 3)


def rotvec(R):   # [..., 3, 3] -> [..., 3] axis * angle (small and moderate angles)
    c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.acos(c)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    k = torch.where(th > 1e-6, th / (2 * torch.sin(th).clamp_min(1e-12)), 0.5 * torch.ones_like(th))
    return v * k[..., None]


def rodrigues(w):   # [..., 3] -> [..., 3, 3]
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    K = skew(w / th)
    th = th[..., None]
    return torch.eye(3, device=w.device) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


class Sweep:
    """One sweep's cross-sweep matches as tensors: the moving side (m) and the fixed side (f)."""

    def __init__(self, M, photos, moves, centres, sid, s, keep, max_matches, dev):
        sel = (sid[M.p[0]] == s) | (sid[M.p[1]] == s)
        if keep is not None:
            sel &= keep
        ids = np.flatnonzero(sel)
        if len(ids) > max_matches:
            ids = np.random.default_rng(0).choice(ids, max_matches, replace=False)
        self.n = len(ids)
        if self.n == 0:
            return
        flip = sid[M.p[0][ids]] != s
        side = lambda arr0, arr1: np.where(flip[:, None] if arr0.ndim > 1 else flip, arr1[ids], arr0[ids])
        pm, pf = np.where(flip, M.p[1][ids], M.p[0][ids]), np.where(flip, M.p[0][ids], M.p[1][ids])
        t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
        Rf, Cf = ms.world(photos, moves, centres, sid, pf)
        self.Cf, self.Rf = t(Cf), t(Rf)
        self.df = torch.einsum("nij,nj->ni", self.Rf, t(side(M.dc[1], M.dc[0])))
        self.R0, self.C0 = t(photos["R"][pm]), t(photos["C"][pm])
        self.dcm = t(side(M.dc[0], M.dc[1]))
        self.xm, self.xf = t(side(M.x[0], M.x[1])), t(side(M.x[1], M.x[0]))
        self.Km, self.Kf = t(photos["K"][pm]), t(photos["K"][pf])
        self.c = t(centres[s])

    def moved(self, R, t):
        """Sweep-side camera centres and rotations under move(s) R [..., 3, 3], t [..., 3]."""
        Cm = self.c + torch.einsum("...ij,nj->...ni", R, self.C0 - self.c) + t[..., None, :]
        Rw = R[..., None, :, :] @ self.R0
        return Cm, Rw

    def errors(self, R, t):
        """Reprojection miss (px, worst of the two photos) and depth from the sweep camera, per
        match, for move(s) R, t (leading batch dims allowed)."""
        Cm, Rw = self.moved(R, t)
        dm = torch.einsum("...nij,nj->...ni", Rw, self.dcm)
        w0 = Cm - self.Cf
        b = (dm * self.df).sum(-1)
        d, e = (dm * w0).sum(-1), (self.df * w0).sum(-1)
        den = (1 - b * b).clamp_min(1e-9)
        s1, s2 = (b * e - d) / den, (e - b * d) / den
        X = 0.5 * (Cm + s1[..., None] * dm + self.Cf + s2[..., None] * self.df)
        def proj(Rc, C, K, x):
            Xc = torch.einsum("...nji,...nj->...ni", Rc, X - C)
            z = Xc[..., 2].clamp_min(1e-6)
            return torch.hypot(K[..., 0] * Xc[..., 0] / z + K[..., 2] - x[..., 0],
                               K[..., 1] * Xc[..., 1] / z + K[..., 3] - x[..., 1]), Xc[..., 2]
        em, zm = proj(Rw, Cm, self.Km, self.xm)
        ef, _ = proj(self.Rf.expand_as(Rw), self.Cf.expand_as(Cm), self.Kf, self.xf)
        bad = (den < 1e-4) | (s1 < 0.03) | (s2 < 0.03)
        return torch.where(bad, torch.full_like(em, float("inf")), torch.maximum(em, ef)), zm

    def turn_information(self, R, t, use):
        """The turn block of the Gauss-Newton matrix (track points eliminated), [3, 3]."""
        H = self.gn_step(R[None], t[None], torch.ones(1, self.n, device=R.device), use[None].float(),
                         torch.ones(1, 1, 1, 6, device=R.device), return_h=True)[0]
        return H[:3, :3]

    def gn_step(self, R, t, w, use, mask, prior=0.0, return_h=False):
        """One Gauss-Newton step for the parameters in mask (turn xyz, shift xyz) from move R, t,
        on matches `use` with weights w; for a batch of hypotheses (leading dim). Point-to-ray
        residuals scaled to pixels (f / distance), track points eliminated per match. prior: a
        pull of the sweep's total turn toward zero, in the same units (see --turn-prior)."""
        Cm, Rw = self.moved(R, t)
        dm = torch.einsum("...nij,nj->...ni", Rw, self.dcm)
        df, Cf = self.df.expand_as(dm), self.Cf.expand_as(Cm)
        # triangulate (least squares over the two rays)
        P = lambda d: torch.eye(3, device=d.device) - d[..., :, None] * d[..., None, :]
        A = P(dm) + P(df)
        X = torch.linalg.solve(A + 1e-9 * torch.eye(3, device=A.device), (P(dm) @ Cm[..., None] + P(df) @ Cf[..., None]))[..., 0]
        fm, ff = self.Km[..., 0], self.Kf[..., 0]
        sm = (fm / (X - Cm).norm(dim=-1).clamp_min(1e-3))[..., None]
        sf = (ff / (X - Cf).norm(dim=-1).clamp_min(1e-3))[..., None]
        rm, rf = torch.cross(dm, X - Cm, dim=-1) * sm, torch.cross(df, X - Cf, dim=-1) * sf
        Jxm, Jxf = skew(dm) * sm[..., None], skew(df) * sf[..., None]
        v, wv = X - Cm, Cm - self.c
        I = torch.eye(3, device=X.device)
        dv = (dm * v).sum(-1)[..., None, None]
        dwv = (dm * wv).sum(-1)[..., None, None]
        Jw = (dm[..., :, None] * v[..., None, :] - dv * I) - (dwv * I - wv[..., :, None] * dm[..., None, :])
        Jp = torch.cat([Jw, -skew(dm)], -1) * sm[..., None]                       # [..., n, 3, 6]
        Jp = Jp * mask
        ww = (w * use)[..., None, None]
        Hxx = Jxm.transpose(-1, -2) @ Jxm + Jxf.transpose(-1, -2) @ Jxf + 1e-9 * I
        Hxp = Jxm.transpose(-1, -2) @ Jp
        gx = (Jxm.transpose(-1, -2) @ rm[..., None] + Jxf.transpose(-1, -2) @ rf[..., None])
        Hi = torch.linalg.inv(Hxx)
        H = (ww * (Jp.transpose(-1, -2) @ Jp - Hxp.transpose(-1, -2) @ Hi @ Hxp)).sum(-3)
        g = (ww * (Jp.transpose(-1, -2) @ rm[..., None] - Hxp.transpose(-1, -2) @ Hi @ gx)).sum(-3)[..., 0]
        if return_h:
            return H
        mv = mask.reshape(6)
        if prior > 0 and mv[:3].sum() > 0:   # damp the turn: strong where the matches pin it weakly
            H = H.clone()
            H[..., :3, :3] += prior * torch.eye(3, device=H.device)
            g = g.clone()
            g[..., :3] += prior * rotvec(R)
        H = H + torch.diag_embed((1 - mv).expand_as(g)) + 1e-6 * torch.eye(6, device=H.device)
        step = -torch.linalg.solve(H, g[..., None])[..., 0] * mv
        dR = rodrigues(step[..., :3])
        return dR @ R, (dR @ t[..., None])[..., 0] + step[..., 3:]


def ransac(sw, R, t, subset, mask, tol, iters, dev, gen, prior=0.0):
    """Best move (only the masked parameters change) from random 3-match samples of `subset`,
    by how many of `subset` then reproject within tol px; refined on its inliers."""
    idx = subset.nonzero(as_tuple=True)[0]
    if len(idx) < 6:
        return R, t, 0
    samp = idx[torch.randint(len(idx), (iters, 3), generator=gen, device="cpu").to(dev)]
    use = torch.zeros(iters, sw.n, device=dev)
    use.scatter_(1, samp, 1.0)
    Rb, tb = R.expand(iters, 3, 3).contiguous(), t.expand(iters, 3).contiguous()
    for _ in range(2):
        Rb, tb = sw.gn_step(Rb, tb, torch.ones(1, sw.n, device=dev), use, mask)
    e, _ = sw.errors(Rb, tb)
    score = ((e < tol) & subset).sum(-1)
    k = int(score.argmax())
    Rk, tk = Rb[k], tb[k]
    for _ in range(4):   # refine on its inliers, Huber-weighted
        e, _ = sw.errors(Rk, tk)
        inl = (e < tol) & subset
        w = torch.where(e < tol / 2, torch.ones_like(e), (tol / 2) / e.clamp_min(1e-6))
        Rk, tk = sw.gn_step(Rk[None], tk[None], w[None], inl[None].float(), mask, prior)
        Rk, tk = Rk[0], tk[0]
    e, _ = sw.errors(Rk, tk)
    return Rk, tk, int(((e < tol) & subset).sum())


def solve_sweep(sw, R, t, a, dev, gen):
    e0, z = sw.errors(R, t)
    ok = torch.isfinite(e0)
    if ok.sum() < 20:
        return None
    zq = z[ok]
    lo, hi = torch.quantile(zq, 1 / 3), torch.quantile(zq, 2 / 3)
    far, near = ok & (z >= hi), ok & (z <= lo)
    ratio = float(z[far].median() / z[near].median().clamp_min(1e-6)) if far.any() and near.any() else 1.0
    f = float(sw.Km[:, 0].mean())
    prior = a.turn_prior * f * f   # the turn's pull toward zero: worth --turn-prior matches' information
    turn_m = torch.tensor([1., 1, 1, 0, 0, 0], device=dev).view(1, 1, 1, 6)
    shift_m = torch.tensor([0., 0, 0, 1, 1, 1], device=dev).view(1, 1, 1, 6)
    both_m = torch.ones(1, 1, 1, 6, device=dev)
    R_in, t_in = R, t
    log = []
    for tol in a.tols:   # far matches order the turn, near ones the shift; overlap is fine
        R, t, nf = ransac(sw, R, t, far, turn_m, tol, a.iters, dev, gen, prior)
        R, t, nn = ransac(sw, R, t, near, shift_m, tol, a.iters, dev, gen, prior)
        log.append((tol, nf, nn))
    tol = a.tols[-1]
    for _ in range(6):   # joint robust fit of all six on every match within the final tolerance
        e, _ = sw.errors(R, t)
        inl = (e < tol) & ok
        w = torch.where(e < tol / 2, torch.ones_like(e), (tol / 2) / e.clamp_min(1e-6))
        R1, t1 = sw.gn_step(R[None], t[None], w[None], inl[None].float(), both_m, prior)
        R, t = R1[0], t1[0]
    e1, _ = sw.errors(R, t)
    before, after = float((e0 < 2 * tol).float().mean()), float((e1 < 2 * tol).float().mean())
    kept = after >= before
    if not kept:   # the move made the matches agree less: keep what the sweep had
        R, t = R_in, t_in
    # how well the matches pin each turn axis, against the prior (>1: the data decides)
    Ht = sw.turn_information(R, t, (e1 < tol) & ok)
    pinned = (torch.linalg.eigvalsh(Ht) / prior).tolist() if prior > 0 else None
    return R, t, {"matches": sw.n, "depth_ratio": round(ratio, 2), "kept": kept,
                  "turn_information": [round(x, 1) for x in pinned] if pinned else None,
                  "agreeing": [round(before, 3), round(after, 3)], "rounds": log}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("database", nargs="+", help="one or more COLMAP databases (repeated runs are pooled)")
    p.add_argument("--poses", required=True)
    p.add_argument("--raw", action="store_true",
                   help="LightGlue's matches before COLMAP's RANSAC verification (the same every run)")
    p.add_argument("--out", required=True)
    p.add_argument("--exclude-names")
    p.add_argument("--box")
    p.add_argument("--box-margin", type=float, default=0.02)
    p.add_argument("--tols", type=float, nargs="+", default=[24.0, 8.0, 3.0], help="px, loosest first")
    p.add_argument("--iters", type=int, default=256, help="RANSAC hypotheses per stage")
    p.add_argument("--turn-prior", type=float, default=10.0,
                   help="pull of each sweep's turn toward zero, as worth this many matches: decides "
                        "only the turn axes the matches pin weakly (default 10)")
    p.add_argument("--rounds", type=int, default=2, help="passes over all sweeps")
    p.add_argument("--max-matches", type=int, default=30000)
    p.add_argument("--sweep-jump", type=float, default=0.0)
    p.add_argument("--sweep-max", type=int, default=0)
    p.add_argument("--sweep-min", type=int, default=3)
    p.add_argument("--write-model")
    p.add_argument("--write-from")
    a = p.parse_args(argv)
    t_start = time.time()
    dev = DEV
    gen = torch.Generator().manual_seed(0)

    stem = lambda n: Path(n).stem
    drop = {stem(l.strip()) for l in open(a.exclude_names)} if a.exclude_names else set()
    images = [im for im in gsdata.read_model(Path(a.poses))[0] if stem(im["name"]) not in drop]
    c2w = np.linalg.inv(np.stack([im["viewmat"] for im in images]))
    names = [im["name"] for im in images]
    sid = np.array(T.find_sweeps(names, torch.tensor(c2w, dtype=torch.float32), a.sweep_jump,
                                 longest=a.sweep_max, min_size=a.sweep_min))
    photos = {"name": names, "R": c2w[:, :3, :3], "C": c2w[:, :3, 3],
              "K": np.array([[im["cam"][k] for k in ("fx", "fy", "cx", "cy")] for im in images])}
    M = ms.Matches(a.database, photos, sid, a.max_matches, raw=a.raw)
    sweeps = sorted(set(sid))
    anchor = max(sweeps, key=lambda s: (sid == s).sum())
    centres = {s: photos["C"][sid == s].mean(0) for s in sweeps}
    moves = {s: (np.eye(3), np.zeros(3)) for s in sweeps}
    keep = None
    if a.box:   # same box filter as match_sweeps
        b = json.loads(Path(a.box).read_text())
        bc, ax, half = np.array(b["center"]), np.array(b["axes"]), np.array(b["half"]) + a.box_margin
        Ra, Ca = ms.world(photos, moves, centres, sid, M.p[0])
        Rb, Cb = ms.world(photos, moves, centres, sid, M.p[1])
        da, db = np.einsum("nij,nj->ni", Ra, M.dc[0]), np.einsum("nij,nj->ni", Rb, M.dc[1])
        w0 = Ca - Cb
        bb = (da * db).sum(-1)
        s1 = (bb * (db * w0).sum(-1) - (da * w0).sum(-1)) / np.maximum(1 - bb * bb, 1e-9)
        keep = (np.abs((Ca + s1[:, None] * da - bc) @ ax.T) <= half).all(1)
    print(f"{len(M.p[0]):,} verified matches between {len(sweeps)} sweeps on {dev}; anchor = sweep {anchor}", flush=True)
    order = sorted((s for s in sweeps if s != anchor), key=lambda s: -(sid == s).sum())
    info = {}
    for rnd in range(a.rounds):
        for s in order:
            sw = Sweep(M, photos, moves, centres, sid, s, keep, a.max_matches, dev)
            if sw.n < 20:
                continue
            R0 = torch.tensor(moves[s][0], dtype=torch.float32, device=dev)
            t0 = torch.tensor(moves[s][1], dtype=torch.float32, device=dev)
            r = solve_sweep(sw, R0, t0, a, dev, gen)
            if r is not None:
                moves[s] = (r[0].cpu().double().numpy(), r[1].cpu().double().numpy())
                info[s] = r[2]
    turn_deg = lambda R: float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    report = {"anchor": int(anchor), "device": dev, "seconds": round(time.time() - t_start, 1), "sweeps": {}}
    for s in sweeps:
        members = sorted(n for n, q in zip(names, sid) if q == s)
        R, t = moves[s]
        inf = info.get(s, {})
        report["sweeps"][str(s)] = {
            "photos": len(members), "first": members[0], "last": members[-1], "anchor": bool(s == anchor),
            "shift_mm": [round(1000 * float(x), 2) for x in t], "shift_size_mm": round(1000 * float(np.linalg.norm(t)), 2),
            "turn_deg": round(turn_deg(R), 3), "R": R.round(8).tolist(), "centre": centres[s].tolist(), **inf}
        t0, t1 = members[0].split("_")[2], members[-1].split("_")[2]
        print(f"  {t0[:2]}:{t0[2:4]}:{t0[4:]}-{t1[2:4]}:{t1[4:]} ({len(members):3d} photos): " +
              ("anchor" if s == anchor else
               f"shift {1000 * np.linalg.norm(t):5.1f} mm, turn {turn_deg(R):.2f} deg; depth ratio "
               f"{inf.get('depth_ratio', 0):.2f}; turn pinned x{min(inf.get('turn_information') or [0]):.0f}"
               f"{'' if inf.get('kept', True) else ' (move rejected)'}; "
               f"agreeing {100 * inf.get('agreeing', [0, 0])[0]:.0f}% -> {100 * inf.get('agreeing', [0, 0])[1]:.0f}%"
               if inf else "too few matches"), flush=True)
    print(f"{time.time() - t_start:.1f} s")
    Path(a.out).write_text(json.dumps(report, indent=1))
    if a.write_model:
        rs.write_moved(Path(a.write_from or a.poses), Path(a.write_model), moves, centres,
                       {stem(n): s for n, s in zip(names, sid)}, anchor)


if __name__ == "__main__":
    main()
