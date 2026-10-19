"""How blurred each photo is, from the width of its strongest edges (numpy + Pillow only).

  python frame_check.py --images <dir> --out widths.json [names...]

For each photo: lightly denoised (so sensor noise and JPEG blocks don't count as detail), the 400
strongest edge points (local maxima of the gradient), and across each one the distance over which
brightness goes from 10% to 90% of the step, in pixels at full resolution. Sharp phone photos come out
around 2.5-4 px; defocus widens every direction, motion blur mostly one. Writes {name: {"width_px",
"by_direction_px": [0, 45, 90, 135 deg]}}; width_px is null for photos with too few edges to judge.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _blur(g, sigma=0.8):
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    g = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 1, g)
    return np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, g)


def _sample(g, x, y):
    """Bilinear samples of g at float coordinates (edge-clamped)."""
    H, W = g.shape
    x = np.clip(x, 0, W - 1.001)
    y = np.clip(y, 0, H - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    ax, ay = x - x0, y - y0
    return (g[y0, x0] * (1 - ax) * (1 - ay) + g[y0, x0 + 1] * ax * (1 - ay)
            + g[y0 + 1, x0] * (1 - ax) * ay + g[y0 + 1, x0 + 1] * ax * ay)


def edge_width(path, n=400, half=12, min_step=20.0):
    g = np.asarray(Image.open(path).convert("L"), np.float32)
    g = _blur(g)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[1:-1, 1:-1] = (g[1:-1, 2:] - g[1:-1, :-2]) * 2 + (g[:-2, 2:] - g[:-2, :-2]) + (g[2:, 2:] - g[2:, :-2])
    gy[1:-1, 1:-1] = (g[2:, 1:-1] - g[:-2, 1:-1]) * 2 + (g[2:, :-2] - g[:-2, :-2]) + (g[2:, 2:] - g[:-2, 2:])
    mag = np.hypot(gx, gy)
    mx = mag.copy()   # 9x9 max filter: one point per edge
    for d in range(1, 5):
        for a in (mx,):
            a[d:] = np.maximum(a[d:], mag[:-d]); a[:-d] = np.maximum(a[:-d], mag[d:])
    mx2 = mx.copy()
    for d in range(1, 5):
        mx2[:, d:] = np.maximum(mx2[:, d:], mx[:, :-d]); mx2[:, :-d] = np.maximum(mx2[:, :-d], mx[:, d:])
    cand = (mag == mx2) & (mag > np.percentile(mag, 99))
    cand[:half + 1] = cand[-half - 1:] = False
    cand[:, :half + 1] = cand[:, -half - 1:] = False
    ys, xs = np.nonzero(cand)
    order = np.argsort(-mag[ys, xs])[:n]
    ys, xs = ys[order], xs[order]
    if len(ys) < 20:
        return None, [None] * 4
    ang = np.arctan2(gy[ys, xs], gx[ys, xs])
    t = np.linspace(-half, half, 4 * half + 1)
    prof = _sample(g, xs[:, None] + t[None] * np.cos(ang)[:, None], ys[:, None] + t[None] * np.sin(ang)[:, None])
    lo, hi = prof[:, :5].mean(1), prof[:, -5:].mean(1)
    step = hi - lo
    ok = np.abs(step) > min_step
    f = np.clip((prof - lo[:, None]) / np.where(np.abs(step) > 1e-6, step, 1)[:, None], 0, 1)
    f = np.maximum.accumulate(f, axis=1)
    widths = np.where(ok, ((f < 0.9).sum(1) - (f < 0.1).sum(1)) * (t[1] - t[0]), np.nan)
    d = np.degrees(ang) % 180
    by = []
    for a in (0, 45, 90, 135):
        w = widths[(d >= a) & (d < a + 45)]
        w = w[~np.isnan(w)]
        by.append(round(float(np.median(w)), 2) if len(w) >= 15 else None)
    w = widths[~np.isnan(widths)]
    return (round(float(np.median(w)), 2) if len(w) >= 20 else None), by


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("names", nargs="*", help="photos to check (default: every .jpg/.png in --images)")
    a = p.parse_args(argv)
    names = a.names or sorted(f.name for f in Path(a.images).iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    res = {}
    for n in names:
        try:
            w, by = edge_width(Path(a.images) / n)
        except Exception as e:
            w, by = None, [None] * 4
        res[n] = {"width_px": w, "by_direction_px": by}
    Path(a.out).write_text(json.dumps(res, indent=0))
    print(f"{len(res)} photos checked", flush=True)


if __name__ == "__main__":
    main()
