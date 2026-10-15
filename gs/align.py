"""Align photos to each other over a region by projecting them onto the LiDAR surface.

  python align.py <model dir> --images <photos> --depth-dir <sensor_data> --box CX CY CZ SX SY SZ
                  [--frame meta.json --viewer-unflips] [--iterations 4] [--step 0.5]
                  [--neighbours 8] --out <model dir for the aligned poses>

Each iteration, every photo's region pixels are lifted to 3D with its own LiDAR depth and
pose; each of its nearest neighbours (nearest in space and viewing direction - never by capture
order) is warped into its view through that surface, and the shift between the photo and the
warp is measured (phase correlation on image edges). The median shift over the neighbours is
how far this photo's view disagrees with theirs; the camera is turned by --step of it (a small
rotation about its own axes, the centre kept). Photos that agree barely move; outliers are
pulled towards where the others see the region. Writes images.txt/cameras.txt to --out.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import box as boxmod
import gsdata

W, H = 640, 480


def edges(x):
    gy, gx = np.gradient(x)
    return np.hypot(gx, gy)


def shift_vec(a, b, m):
    """(dx, dy) in working pixels: where the region in b sits relative to a (subpixel)."""
    ys, xs = np.nonzero(m)
    if len(ys) < 400:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if (y1 - y0) < 24 or (x1 - x0) < 24:
        return None
    a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    ea, eb = edges(a), edges(b)
    A, B = np.fft.fft2((ea - ea.mean()) * win), np.fft.fft2((eb - eb.mean()) * win)
    X = A * np.conj(B)
    X /= np.abs(X) + 1e-9
    c = np.fft.ifft2(X).real
    h, w = c.shape
    i = np.unravel_index(np.argmax(c), c.shape)

    def sub(axis):
        k, n = i[axis], c.shape[axis]
        m1 = c[(i[0] - 1) % h, i[1]] if axis == 0 else c[i[0], (i[1] - 1) % w]
        p1 = c[(i[0] + 1) % h, i[1]] if axis == 0 else c[i[0], (i[1] + 1) % w]
        v = k + (m1 - p1) / (2 * (m1 - 2 * c[i] + p1) + 1e-12)
        return v - n if v > n / 2 else v
    return np.array([sub(1), sub(0)])


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Photo:
    def __init__(self, im, images, depth_dir, center, axes, half):
        self.im = im
        cam = im["cam"]
        self.g = np.asarray(Image.open(Path(images) / im["name"]).convert("L").resize((W, H), Image.BILINEAR), float) / 255
        d, _ = gsdata.read_depth(depth_dir, im["name"])
        self.z = np.asarray(Image.fromarray(d.numpy()).resize((W, H), Image.NEAREST))
        sx, sy = W / cam["w"], H / cam["h"]
        self.K = (cam["fx"] * sx, cam["fy"] * sy, cam["cx"] * sx, cam["cy"] * sy)
        self.R, self.t = im["viewmat"][:3, :3].copy(), im["viewmat"][:3, 3].copy()
        self.box = (center, axes, half)
        self.update()

    def update(self):
        center, axes, half = self.box
        v, u = np.mgrid[0:H, 0:W]
        K, z = self.K, self.z
        pc = np.stack([(u + .5 - K[2]) / K[0] * z, (v + .5 - K[3]) / K[1] * z, z], -1)
        self.Pw = (pc - self.t) @ self.R
        self.inside = (np.abs((self.Pw - center) @ axes) <= half).all(-1) & (z > 0.05)
        self.C = -self.R.T @ self.t

    def warp_from(self, o):
        q = self.Pw @ o.R.T + o.t
        ok = self.inside & (q[..., 2] > 0.05)
        uu = o.K[0] * q[..., 0] / np.maximum(q[..., 2], 1e-6) + o.K[2] - .5
        vv = o.K[1] * q[..., 1] / np.maximum(q[..., 2], 1e-6) + o.K[3] - .5
        ok &= (uu >= 0) & (uu < W - 1) & (vv >= 0) & (vv < H - 1)
        ui, vi = np.clip(uu.astype(int), 0, W - 2), np.clip(vv.astype(int), 0, H - 2)
        fu, fv = np.clip(uu - ui, 0, 1), np.clip(vv - vi, 0, 1)
        g = o.g
        w = (g[vi, ui] * (1 - fu) * (1 - fv) + g[vi, ui + 1] * fu * (1 - fv)
             + g[vi + 1, ui] * (1 - fu) * fv + g[vi + 1, ui + 1] * fu * fv)
        return np.where(ok, w, self.g), ok

    def turn(self, dx, dy):
        """Rotate the camera (centre fixed) so its image content moves by (dx, dy) working px."""
        ax, ay = math.atan2(dy, self.K[1]), math.atan2(-dx, self.K[0])
        Rd = rot("x", ax) @ rot("y", ay)
        C = self.C
        self.R = Rd @ self.R
        self.t = -self.R @ C
        self.update()


def neighbours(p, photos, k):
    """Nearest in space and viewing direction (never by capture order), seeing the region."""
    cand = [o for o in photos if o is not p and o.inside.sum() > 400]
    f = p.R[2]
    cand.sort(key=lambda o: np.linalg.norm(o.C - p.C) + 0.2 * (1 - f @ o.R[2]))
    return cand[:k]


def measure(photos, k, agree=0.0):
    """Median shift per photo over its neighbours. With agree > 0, photos whose neighbours'
    shifts scatter more than that fraction of the median shift are left out (no consensus)."""
    out = {}
    for p in photos:
        if p.inside.sum() < 400:
            continue
        vs = []
        for o in neighbours(p, photos, k):
            w, ok = p.warp_from(o)
            s = shift_vec(p.g, w, ok)
            if s is not None and np.hypot(*s) < W / 4:
                vs.append(s)
        if len(vs) >= 3:
            vs = np.array(vs)
            m = np.median(vs, axis=0)
            spread = np.median(np.linalg.norm(vs - m, axis=1))
            if agree <= 0 or spread <= agree * max(np.linalg.norm(m), 2.0):
                out[p.im["name"]] = m
    return out


def write_model(src, photos, out):
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(src) / "cameras.txt", out / "cameras.txt")
    by = {p.im["name"]: p for p in photos}
    lines = [l.rstrip("\n") for l in open(Path(src) / "images.txt") if not l.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    res = []
    for k in range(0, len(lines), 2):
        p = lines[k].split()
        if len(p) >= 10 and p[9] in by:
            ph = by[p[9]]
            q = boxmod.rotmat_to_qvec(ph.R)
            p[1:5] = [f"{x:.12f}" for x in q]
            p[5:8] = [f"{x:.12f}" for x in ph.t]
        res += [" ".join(p), lines[k + 1] if k + 1 < len(lines) else ""]
    (out / "images.txt").write_text("\n".join(res) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--images", required=True)
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--box", type=float, nargs=6, required=True)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--frame")
    p.add_argument("--viewer-unflips", action="store_true")
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--neighbours", type=int, default=8)
    p.add_argument("--max-turn", type=float, default=2.0, help="largest total correction per photo, degrees")
    p.add_argument("--agree", type=float, default=1.0,
                   help="only turn a photo when its neighbours' shifts scatter less than this x their median")
    p.add_argument("--report", help="score-only: write per-photo disagreement JSON and stop")
    p.add_argument("--out")
    a = p.parse_args(argv)
    meta = json.loads(Path(a.frame).read_text()) if a.frame else None
    center, axes, half = boxmod.box_in_training(a.box[:3], a.box[3:], a.yaw, meta, a.viewer_unflips)
    photos = [Photo(im, a.images, a.depth_dir, center, axes, half) for im in gsdata.read_model(Path(a.model))[0]]
    scale = 1920 / W
    if a.report:   # just score
        sh = measure(photos, a.neighbours)
        mags = {n: float(np.hypot(*v) * scale) for n, v in sh.items()}
        m = np.array(list(mags.values()))
        print(f"{len(m)} photos: disagreement median {np.median(m):.1f} px, p90 {np.percentile(m, 90):.1f} px, "
              f">10 px: {int((m > 10).sum())}, >20 px: {int((m > 20).sum())}")
        Path(a.report).write_text(json.dumps(mags, indent=1))
        return
    turned = {ph.im["name"]: np.zeros(2) for ph in photos}
    cap = math.radians(a.max_turn)
    for it in range(a.iterations + 1):
        sh = measure(photos, a.neighbours, a.agree if it < a.iterations else 0.0)
        mags = np.array([np.hypot(*v) for v in sh.values()]) * scale
        print(f"iteration {it}: disagreement median {np.median(mags):.1f} px, p90 {np.percentile(mags, 90):.1f} px, "
              f">10 px: {int((mags > 10).sum())}, >20 px: {int((mags > 20).sum())}", flush=True)
        if it == a.iterations:
            break
        for ph in photos:
            v = sh.get(ph.im["name"])
            if v is None:
                continue
            d = -a.step * v
            tot = turned[ph.im["name"]] + d
            lim = math.tan(cap) * ph.K[0]   # the cap, in working pixels
            if np.linalg.norm(tot) > lim:   # clip the total correction to the cap
                d = tot * (lim / np.linalg.norm(tot)) - turned[ph.im["name"]]
            ph.turn(*d)
            turned[ph.im["name"]] += d
    write_model(a.model, photos, Path(a.out))
    tm = np.array([np.hypot(*v) for v in turned.values()]) * scale
    print(f"cameras turned by: median {np.median(tm):.1f} px-equivalent, max {tm.max():.1f}; written to {a.out}")


if __name__ == "__main__":
    main()
