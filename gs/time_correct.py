"""Smooth time correction: a small correction for every photo's camera, smooth along the capture.

  python time_correct.py <database.db> [more.db ...] --poses <model> --out <result.json>
                         [--raw] [--render <names.txt>=<search.json> ...] [--box <box.json>]
                         [--write-model <dir> [--write-from <model>]]

Sweep correction moves whole sweeps rigidly; tracking drift isn't rigid (it builds up along a
sweep, faster when the phone moves fast or turns sharply). Here each photo gets its own turn
(about its camera centre) and shift, and the corrections are tied together in capture order:
consecutive photos may differ by a little plus a share of how far the phone moved between them,
so the correction bends where the motion was violent and stays put where it was calm.

What pins the corrections down:
  - matches between photos (LightGlue raw or verified, from COLMAP's database). Band filter
    first: a match must already reproject within --band px under the ARKit poses (true matches
    can't be further off than the drift), in front of both cameras, nearer than --max-depth;
    repeated objects (a cup matched to the next cup) mostly fall outside it. Robust loss on the
    rest, then a second solve without photo pairs whose matches mostly disagree.
  - optionally, render-search results for groups of photos (sweep_search.py's result json, with
    the group's photo list): the group's mean correction is pulled toward the searched move. For
    stretches no reliable match reaches.
  - optionally, LiDAR (--lidar <sensor_data dir>): each photo's high-confidence depth points,
    placed with its corrected camera, should lie on the LiDAR surfaces of photos taken more than
    --far-seconds away (point to plane, robust). The depth was measured in the camera's own frame,
    so it checks the pose directly, matches or not, and glass doesn't fool it (no confident return).
    Correspondences are refreshed between solves, with a shrinking reach.
  - the largest sweep is the anchor and stays fixed; a weak pull toward zero everywhere.

Reported per photo: the correction, and its "far links": agreeing matches to photos taken more
than --far-seconds away. Matches to the previous few photos only chain a photo to its
neighbours, which share its drift; a photo with no far links can only be placed by smoothness or
a render search, and is flagged. Runs on CUDA, Apple MPS or CPU.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

import box as boxmod
import gsdata
import match_sweeps as ms
import train as T

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def rodrigues(w):   # [N, 3] radians -> [N, 3, 3]; I + A [w]x + B [w]x^2, smooth through zero
    th2 = (w * w).sum(-1, keepdim=True)[..., None]
    small = th2 < 1e-8
    th = th2.clamp_min(1e-8).sqrt()
    A = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    B = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-8))
    z = torch.zeros_like(w[:, 0])
    W = torch.stack([z, -w[:, 2], w[:, 1], w[:, 2], z, -w[:, 0], -w[:, 1], w[:, 0], z], -1).view(-1, 3, 3)
    return torch.eye(3, dtype=w.dtype, device=w.device) + A * W + B * W @ W


def rv_matrix(w_deg):
    """Rotation matrix of a rotation vector in degrees (numpy)."""
    w = np.radians(np.asarray(w_deg, float))
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def capture_seconds(names):
    """Seconds since midnight from names like wide_YYYYMMDD_HHMMSS_xxx (else the photo index)."""
    out = []
    for k, n in enumerate(names):
        m = re.search(r"_(\d{8})_(\d{2})(\d{2})(\d{2})", Path(n).stem)
        out.append(int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4]) if m else float(k))
    return np.array(out, float)


class Problem:
    def __init__(self, photos, M, dev, dtype):
        t = lambda x: torch.tensor(np.asarray(x), dtype=dtype, device=dev)
        self.R0, self.C0, self.K = t(photos["R"]), t(photos["C"]), t(photos["K"])
        self.pa = torch.tensor(M.p[0], device=dev)
        self.pb = torch.tensor(M.p[1], device=dev)
        self.da, self.db = t(M.dc[0]), t(M.dc[1])
        self.xa, self.xb = t(M.x[0]), t(M.x[1])

    def cameras(self, P):
        """P [N, 6]: turn (deg, world axes, about the camera centre) and shift (mm)."""
        R = rodrigues(torch.deg2rad(P[:, :3])) @ self.R0
        return R, self.C0 + P[:, 3:] / 1000

    def misses(self, P, sel=None):
        """Per match: reprojection miss in photo a and b (px), distance along each ray (m), depth in a."""
        R, C = self.cameras(P)
        pa, pb = (self.pa, self.pb) if sel is None else (self.pa[sel], self.pb[sel])
        da0, db0 = (self.da, self.db) if sel is None else (self.da[sel], self.db[sel])
        xa, xb = (self.xa, self.xb) if sel is None else (self.xa[sel], self.xb[sel])
        Ra, Rb, Ca, Cb = R[pa], R[pb], C[pa], C[pb]
        da = torch.einsum("nij,nj->ni", Ra, da0)
        db = torch.einsum("nij,nj->ni", Rb, db0)
        w0 = Ca - Cb
        b = (da * db).sum(-1)
        d, e = (da * w0).sum(-1), (db * w0).sum(-1)
        den = (1 - b * b).clamp_min(1e-9)
        s1, s2 = (b * e - d) / den, (e - b * d) / den
        X = 0.5 * (Ca + s1[:, None] * da + Cb + s2[:, None] * db)

        def proj(Rc, Cc, K, x):
            Xc = torch.einsum("nji,nj->ni", Rc, X - Cc)
            z = Xc[:, 2].clamp_min(1e-6)
            return torch.hypot(K[:, 0] * Xc[:, 0] / z + K[:, 2] - x[:, 0], K[:, 1] * Xc[:, 1] / z + K[:, 3] - x[:, 1]), Xc[:, 2]
        ea, za = proj(Ra, Ca, self.K[pa], xa)
        eb, _ = proj(Rb, Cb, self.K[pb], xb)
        bad = (den < 1e-4) | (s1 < 0.03) | (s2 < 0.03)
        big = torch.full_like(ea, 1e4)
        return torch.where(bad, big, ea), torch.where(bad, big, eb), s1, s2, za


def lidar_points(sensor_dir, stem, per_photo, max_depth, rng, margin=0.08):
    """A photo's confident LiDAR points and normals in its camera frame (COLMAP axes), from the
    original SplatKing depth map and ARKit's intrinsics for that frame. None if unavailable."""
    meta_f = Path(sensor_dir) / f"{stem}.json"
    if not meta_f.exists():
        return None
    meta = json.loads(meta_f.read_text()).get("metadata", {}).get("camera")
    d, c = gsdata.read_depth(sensor_dir, stem)
    if meta is None or d is None:
        return None
    k = meta["intrinsics"]
    fx, fy, cx, cy = k[0], k[4], k[6], k[7]
    d, c = d.numpy().astype(np.float64), c.numpy()
    h, w = d.shape
    sc = meta["imageResolution"]["width"] / w
    v, u = np.mgrid[0:h, 0:w]
    x, y = (u + 0.5) * sc - 0.5, (v + 0.5) * sc - 0.5
    P = np.stack([(x - cx) / fx * d, (y - cy) / fy * d, d], -1)
    # normals from the depth grid's neighbours, where the surface is continuous
    du, dv = P[1:-1, 2:] - P[1:-1, :-2], P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)
    ok = (c[1:-1, 1:-1] == 2) & (d[1:-1, 1:-1] > 0.1) & (d[1:-1, 1:-1] < max_depth)
    for nb in (d[1:-1, 2:], d[1:-1, :-2], d[2:, 1:-1], d[:-2, 1:-1]):
        ok &= np.abs(nb - d[1:-1, 1:-1]) < 0.01
    uu, vv = u[1:-1, 1:-1], v[1:-1, 1:-1]
    ok &= (uu >= margin * w) & (uu < (1 - margin) * w) & (vv >= margin * h) & (vv < (1 - margin) * h)
    idx = np.flatnonzero(ok)
    if len(idx) < 50:
        return None
    idx = rng.choice(idx, min(per_photo, len(idx)), replace=False)
    return P[1:-1, 1:-1].reshape(-1, 3)[idx], n.reshape(-1, 3)[idx]


class Lidar:
    """Every photo's LiDAR points (camera frame), matched point-to-plane to other photos' points."""

    def __init__(self, sensor_dir, names, secs, per_photo, max_depth, dev, dtype):
        rng = np.random.default_rng(0)
        loc, nrm, ph = [], [], []
        for i, n in enumerate(names):
            r = lidar_points(sensor_dir, Path(n).stem, per_photo, max_depth, rng)
            if r is not None:
                loc.append(r[0]); nrm.append(r[1]); ph.append(np.full(len(r[0]), i))
        self.photos_with = len(loc)
        t = lambda x: torch.tensor(np.concatenate(x), dtype=dtype, device=dev)
        self.loc, self.nrm = t(loc), t(nrm)
        self.ph = torch.tensor(np.concatenate(ph), device=dev)
        self.secs = torch.tensor(secs, dtype=dtype, device=dev)[self.ph]
        self.pairs = None

    def world(self, R, C, sel=None):
        ph = self.ph if sel is None else self.ph[sel]
        loc = self.loc if sel is None else self.loc[sel]
        nrm = self.nrm if sel is None else self.nrm[sel]
        return C[ph] + torch.einsum("nij,nj->ni", R[ph], loc), torch.einsum("nij,nj->ni", R[ph], nrm)

    @torch.no_grad()
    def match(self, R, C, reach, far_seconds, chunk=2048):
        """Nearest point of a photo taken more than far_seconds away, within reach, facing the same way."""
        q, nq = self.world(R, C)
        qf = q.float()
        iq, ir = [], []
        for k in range(0, len(q), chunk):
            d = torch.cdist(qf[k:k + chunk], qf)
            d[(self.secs[k:k + chunk, None] - self.secs[None, :]).abs() <= far_seconds] = float("inf")
            dm, im = d.min(1)
            rows = torch.arange(k, min(k + chunk, len(q)), device=q.device)
            ok = (dm < reach) & ((nq[rows] * nq[im]).sum(-1).abs() > 0.7)
            iq.append(rows[ok]); ir.append(im[ok])
        self.pairs = (torch.cat(iq), torch.cat(ir))
        return len(self.pairs[0])

    def residuals(self, R, C):
        """Point-to-plane distance (m) of each matched point to its partner's surface."""
        iq, ir = self.pairs
        q, _ = self.world(R, C, iq)
        p, n = self.world(R, C, ir)
        return ((q - p) * n).sum(-1)

    @torch.no_grad()
    def on_surface(self, R, C, reach, far_seconds, N):
        """Per photo: share of its points within 2 mm of another photo's surface (taken far apart)."""
        saved = self.pairs
        self.match(R, C, reach, far_seconds)
        r = self.residuals(R, C).abs()
        ph = self.ph[self.pairs[0]]
        tot = torch.bincount(self.ph, minlength=N).double()
        good = torch.bincount(ph[r < 0.002], minlength=N).double()
        self.pairs = saved
        return (good / tot.clamp_min(1)).cpu().numpy(), tot.cpu().numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("database", nargs="+")
    p.add_argument("--poses", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--raw", action="store_true", help="LightGlue's matches before COLMAP's verification")
    p.add_argument("--band", type=float, default=30.0, help="px: keep matches that already nearly fit the ARKit poses")
    p.add_argument("--max-depth", type=float, default=3.0, help="m")
    p.add_argument("--robust", type=float, default=3.0, help="px, scale of the robust (Cauchy) match loss")
    p.add_argument("--match-weight", type=float, default=1.0, help="0 leaves the feature matches out")
    p.add_argument("--pair-gate", type=float, default=0.3,
                   help="second solve drops photo pairs with fewer than this share of matches agreeing")
    p.add_argument("--smooth-mm", type=float, default=0.5, help="allowed change between consecutive photos, mm ...")
    p.add_argument("--smooth-share", type=float, default=0.05, help="... plus this share of how far the phone moved")
    p.add_argument("--smooth-deg", type=float, default=0.02, help="allowed turn change between photos, deg ...")
    p.add_argument("--smooth-turn-share", type=float, default=0.02, help="... plus this share of how far it turned")
    p.add_argument("--gap-seconds", type=float, default=5.0, help="no smoothness across a pause this long")
    p.add_argument("--prior-mm", type=float, default=20.0, help="weak pull of every correction toward zero")
    p.add_argument("--prior-deg", type=float, default=1.0)
    p.add_argument("--render", nargs="*", default=[], metavar="NAMES=SEARCH_JSON",
                   help="render-search results: the group's mean correction is pulled to the searched move")
    p.add_argument("--render-mm", type=float, default=1.5)
    p.add_argument("--render-lateral", action="store_true",
                   help="pull each group photo's sideways/vertical shift (camera x, y) and turn to the searched move, "
                        "leaving the view direction to LiDAR and matches (renders pin lateral position, LiDAR depth)")
    p.add_argument("--render-deg", type=float, default=0.15)
    p.add_argument("--lidar", help="SplatKing sensor_data folder (original depth maps + per-frame json)")
    p.add_argument("--lidar-points", type=int, default=1500, help="LiDAR points used per photo")
    p.add_argument("--lidar-sigma", type=float, default=1.0, help="mm: a LiDAR miss this big costs as much as ...")
    p.add_argument("--lidar-robust", type=float, default=2.0, help="mm, scale of the robust LiDAR loss")
    p.add_argument("--lidar-reach", type=float, nargs="+", default=[15.0, 8.0, 4.0],
                   help="mm, correspondence reach per round (one solve per round)")
    p.add_argument("--lidar-max-depth", type=float, default=1.5)
    p.add_argument("--far-seconds", type=float, default=10.0)
    p.add_argument("--far-links", type=int, default=30, help="flag photos with fewer agreeing far matches than this")
    p.add_argument("--box", help="box.json: also report how far each correction moves the box centre, as seen from the photo")
    p.add_argument("--sweep-jump", type=float, default=0.0)
    p.add_argument("--sweep-min", type=int, default=3)
    p.add_argument("--pair-cap", type=int, default=0, help="use at most this many band matches per photo pair (0 = all)")
    p.add_argument("--chunk", type=int, default=1_000_000, help="matches per backward pass (bounds GPU memory)")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--write-model")
    p.add_argument("--write-from")
    a = p.parse_args(argv)
    t_start = time.time()
    dev = DEV
    dtype = torch.float32 if dev == "mps" else torch.float64

    images = gsdata.read_model(Path(a.poses))[0]
    order = np.argsort([Path(im["name"]).stem for im in images], kind="stable")
    images = [images[i] for i in order]
    names = [im["name"] for im in images]
    N = len(names)
    c2w = np.linalg.inv(np.stack([im["viewmat"] for im in images]))
    photos = {"name": names, "R": c2w[:, :3, :3], "C": c2w[:, :3, 3],
              "K": np.array([[im["cam"][k] for k in ("fx", "fy", "cx", "cy")] for im in images])}
    sid = np.array(T.find_sweeps(names, torch.tensor(c2w, dtype=torch.float32), a.sweep_jump, min_size=a.sweep_min))
    anchor = int(np.bincount(sid).argmax())
    secs = capture_seconds(names)

    # every photo pair, not just across sweeps: the matches inside a sweep tie neighbours together
    M = ms.Matches(a.database, photos, np.arange(N), 0, raw=a.raw)
    prob = Problem(photos, M, dev, dtype)
    P0 = torch.zeros(N, 6, dtype=dtype, device=dev)
    keep = []
    for k in range(0, len(prob.pa), 2_000_000):   # in chunks: big captures have tens of millions of raw matches
        ch = torch.arange(k, min(k + 2_000_000, len(prob.pa)), device=dev)
        ea, eb, s1, s2, za = prob.misses(P0, ch)
        keep.append((torch.maximum(ea, eb) < a.band) & (s1 > 0.03) & (s2 > 0.03) & (za < a.max_depth))
    keep = torch.cat(keep)
    print(f"{N} photos, {len(np.unique(sid))} sweeps (anchor {anchor}); {len(M.p[0]):,} matches, "
          f"{int(keep.sum()):,} within the {a.band:g} px band; on {dev}", flush=True)

    # smoothness between consecutive photos, scaled by how far the phone moved and turned
    C, R = photos["C"], photos["R"]
    step_mm = 1000 * np.linalg.norm(np.diff(C, axis=0), axis=1)
    rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
    step_deg = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    linked = np.diff(secs) <= a.gap_seconds
    sig = np.stack([a.smooth_deg + a.smooth_turn_share * step_deg] * 3 + [a.smooth_mm + a.smooth_share * step_mm] * 3, 1)
    sig_t = torch.tensor(sig, dtype=dtype, device=dev)
    link_t = torch.tensor(linked, device=dev)
    prior = torch.tensor([a.prior_deg] * 3 + [a.prior_mm] * 3, dtype=dtype, device=dev)
    free = torch.tensor(sid != anchor, dtype=dtype, device=dev)[:, None]
    lidar = None
    if a.lidar:
        lidar = Lidar(a.lidar, names, secs, a.lidar_points, a.lidar_max_depth, dev, dtype)
        print(f"LiDAR: {len(lidar.loc):,} points from {lidar.photos_with} photos", flush=True)

    # render-search groups -> target mean correction per group
    idx = {Path(n).stem: k for k, n in enumerate(names)}
    groups = []
    for spec in a.render:
        nf, rf = spec.split("=", 1)
        members = [idx[Path(l.strip()).stem] for l in open(nf) if l.strip() and Path(l.strip()).stem in idx]
        res = json.loads(Path(rf).read_text())["result"]
        Rg, cg, tg = rv_matrix(res["turn_deg"]), np.array(res["centre"]), np.array(res["shift_mm"]) / 1000
        tau = np.array([(Rg - np.eye(3)) @ (C[i] - cg) + tg for i in members]) * 1000
        target = np.concatenate([np.array(res["turn_deg"], float), tau.mean(0)])
        per = np.concatenate([np.repeat(np.array(res["turn_deg"], float)[None], len(members), 0), tau], 1)
        groups.append((torch.tensor(members, device=dev), torch.tensor(target, dtype=dtype, device=dev), Path(nf).stem,
                       torch.tensor(per, dtype=dtype, device=dev)))
    gsig = torch.tensor([a.render_deg] * 3 + [a.render_mm] * 3, dtype=dtype, device=dev)
    R0_t = torch.tensor(R, dtype=dtype, device=dev)

    def solve(sel, init=None, use_lidar=False):
        X = (torch.zeros(N, 6, dtype=dtype, device=dev) if init is None else init.clone()).requires_grad_(True)
        opt = torch.optim.LBFGS([X], lr=1, max_iter=a.iters, history_size=50, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)
        c2 = a.robust ** 2

        def closure():
            opt.zero_grad()
            total = 0.0
            if a.match_weight > 0:   # in chunks, each back-propagated on its own: memory stays bounded
                for k in range(0, len(sel), a.chunk):
                    e1, e2, _, _, _ = prob.misses(X * free, sel[k:k + a.chunk])
                    part = a.match_weight * (c2 * (torch.log1p(e1 ** 2 / c2) + torch.log1p(e2 ** 2 / c2))).sum() / c2
                    part.backward()
                    total += float(part)
            P = X * free
            loss = torch.zeros((), dtype=dtype, device=dev)
            d = (P[1:] - P[:-1]) / sig_t
            loss = loss + (d ** 2).sum(-1)[link_t].sum() + ((P / prior) ** 2).sum()
            for m, target, _, per in groups:
                if a.render_lateral:   # per photo: turn, and the shift along the camera's x and y axes
                    d = P[m] - per
                    axes = R0_t[m][:, :, :2]                     # camera x, y in world, [n, 3, 2]
                    lat = torch.einsum("nij,ni->nj", axes, d[:, 3:])
                    loss = loss + ((d[:, :3] / gsig[:3]) ** 2).sum() + ((lat / gsig[3]) ** 2).sum()
                else:
                    loss = loss + (((P[m].mean(0) - target) / gsig) ** 2).sum() * len(m)
            if use_lidar:
                r = lidar.residuals(*prob.cameras(P)) * 1000
                cl = a.lidar_robust ** 2
                loss = loss + (cl * torch.log1p(r ** 2 / cl)).sum() / a.lidar_sigma ** 2
            loss.backward()
            return loss.detach() + total
        opt.step(closure)
        return (X * free).detach()

    sel = torch.nonzero(keep).squeeze(1)
    if a.pair_cap > 0:   # a random subset per photo pair: big captures have millions of matches
        pid = (prob.pa[sel] * N + prob.pb[sel]).cpu().numpy()
        order = np.random.default_rng(0).permutation(len(pid))
        pid_o = pid[order]
        srt = np.argsort(pid_o, kind="stable")
        first = np.r_[0, np.flatnonzero(np.diff(pid_o[srt])) + 1]
        rank = np.empty(len(pid), int)
        rank[srt] = np.arange(len(pid)) - np.repeat(first, np.diff(np.r_[first, len(pid)]))
        sel = sel[torch.tensor(np.sort(order[rank < a.pair_cap]), device=dev)]
        print(f"capped at {a.pair_cap} per photo pair: {len(sel):,} matches", flush=True)
    reach = [x / 1000 for x in a.lidar_reach]
    if lidar is not None:
        n_l = lidar.match(*prob.cameras(P0), reach[0], a.far_seconds)
        print(f"LiDAR round 1: {n_l:,} point pairs within {1000 * reach[0]:g} mm", flush=True)
    P = solve(sel, use_lidar=lidar is not None)
    # second solve without photo pairs whose matches mostly disagree after the first
    e1, e2, _, _, _ = prob.misses(P, sel)
    agree = (torch.maximum(e1, e2) < 2 * a.robust)
    pair = prob.pa[sel] * N + prob.pb[sel]
    upair, inv = torch.unique(pair, return_inverse=True)
    n_all = torch.bincount(inv, minlength=len(upair)).double()
    n_ok = torch.bincount(inv, weights=agree.double(), minlength=len(upair))
    good_pair = (n_ok / n_all >= a.pair_gate) & (n_ok >= 8)
    sel2 = sel[good_pair[inv]]
    print(f"first solve: {int(agree.sum()):,} matches agree; {int(good_pair.sum())} of {len(upair)} photo pairs kept "
          f"({len(sel2):,} matches)", flush=True)
    P = solve(sel2, P, use_lidar=lidar is not None)
    for k, rch in enumerate(reach[1:] if lidar is not None else []):
        n_l = lidar.match(*prob.cameras(P), rch, a.far_seconds)
        print(f"LiDAR round {k + 2}: {n_l:,} point pairs within {1000 * rch:g} mm", flush=True)
        P = solve(sel2, P, use_lidar=True)

    # report
    e1, e2, _, _, _ = prob.misses(P, sel2)
    ok = (torch.maximum(e1, e2) < 2 * a.robust).cpu().numpy()
    pa, pb = M.p[0][sel2.cpu().numpy()], M.p[1][sel2.cpu().numpy()]
    far = np.abs(secs[pa] - secs[pb]) > a.far_seconds
    far_links = np.bincount(pa[ok & far], minlength=N) + np.bincount(pb[ok & far], minlength=N)
    links = np.bincount(pa[ok], minlength=N) + np.bincount(pb[ok], minlength=N)
    Pn = P.cpu().double().numpy()
    Rn, Cn = (x.cpu().double().numpy() for x in prob.cameras(P))
    X = np.array(json.loads(Path(a.box).read_text())["center"]) if a.box else None
    surf0 = surf1 = lid_n = None
    if lidar is not None:
        surf0, lid_n = lidar.on_surface(*prob.cameras(P0), 0.02, a.far_seconds, N)
        surf1, _ = lidar.on_surface(*prob.cameras(P), 0.02, a.far_seconds, N)
    report = {"device": dev, "anchor_sweep": anchor, "photos": {}, "groups": [g[2] for g in groups]}
    print(f"{'photo':>16s} sweep  links   far" + ("  lidar 2mm" if lidar is not None else "") + "   turn deg    shift mm"
          + ("   box centre moves mm" if a.box else ""))
    for i, n in enumerate(names):
        disp = None
        if X is not None:   # where the box centre appears to move, in the photo's original frame
            disp = 1000 * (C[i] + R[i] @ Rn[i].T @ (X - Cn[i]) - X)
        lidar_ok = surf1 is not None and lid_n[i] >= 200 and surf1[i] >= 0.5
        flag = sid[i] != anchor and far_links[i] < a.far_links and not lidar_ok
        report["photos"][Path(n).stem] = {
            "sweep": int(sid[i]), "turn_deg": Pn[i, :3].round(4).tolist(), "shift_mm": Pn[i, 3:].round(3).tolist(),
            "links": int(links[i]), "far_links": int(far_links[i]), "needs_render": bool(flag),
            **({"lidar_on_surface": [round(float(surf0[i]), 3), round(float(surf1[i]), 3)]} if surf1 is not None else {}),
            **({"box_move_mm": disp.round(2).tolist()} if disp is not None else {})}
        print(f"{Path(n).stem[-17:]:>16s} {sid[i]:5d} {links[i]:6d} {far_links[i]:5d}"
              + (f" {100 * surf0[i]:3.0f}->{100 * surf1[i]:3.0f}%" if surf1 is not None else "")
              + f" {np.linalg.norm(Pn[i, :3]):9.3f}"
              f" {np.linalg.norm(Pn[i, 3:]):7.2f} {str(Pn[i, 3:].round(1)):>20s}"
              + (f" {np.linalg.norm(disp):6.1f} {str(disp.round(1))}" if disp is not None else "")
              + ("  anchor" if sid[i] == anchor else "  needs render search" if flag else ""))
    report["seconds"] = round(time.time() - t_start, 1)
    report["needs_render"] = [Path(n).stem for n, v in zip(names, report["photos"].values()) if v["needs_render"]]
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"{len(report['needs_render'])} photos with neither far links nor LiDAR agreement; {report['seconds']:.1f} s",
          flush=True)

    if a.write_model:
        src, out = Path(a.write_from or a.poses), Path(a.write_model)
        out.mkdir(parents=True, exist_ok=True)
        for f in ("cameras.txt", "points3D.txt"):
            if (src / f).exists():
                (out / f).write_bytes((src / f).read_bytes())
        at = {Path(n).stem: i for i, n in enumerate(names)}
        lines = [l.rstrip("\n") for l in open(src / "images.txt")]
        res, k = [], 0
        while k < len(lines):
            f = lines[k].split()
            if lines[k].startswith("#") or len(f) < 10:
                res.append(lines[k])
                k += 1
                continue
            i = at.get(Path(f[9]).stem)
            if i is not None:
                Rw = Rn[i].T
                f[1:5] = [f"{x:.12f}" for x in boxmod.rotmat_to_qvec(Rw)]
                f[5:8] = [f"{x:.12f}" for x in -Rw @ Cn[i]]
            res += [" ".join(f), lines[k + 1] if k + 1 < len(lines) else ""]
            k += 2
        (out / "images.txt").write_text("\n".join(res) + "\n")


if __name__ == "__main__":
    main()
