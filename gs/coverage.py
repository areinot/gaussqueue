"""Per-pixel training weights from how densely each part of each photo samples the surface.

  python coverage.py <data> --out <dir> [--mode relative|absolute] [--cell 0.01] [--floor 0.05]
                     [--power 1] [--min 0.5] [--max 2]

relative: a pixel's density over the best any photo reaches in its surface cell (close-ups win
  where they exist; distant pixels of the same surface step back almost to nothing).
absolute: each depth pixel's patch - the surface area its cone covers, (depth / focal)^2 over
  the cosine of the angle to the surface (normals from the depth map) - gives its sampling
  density; the weight is that density over the median photo's median patch, clamped to
  [--min, --max]. Close parts of any photo count more, far parts less, at patch resolution.

<data> is a box dataset (box.py): images, sparse/0, sensor_data (cropped LiDAR depth), masks.
Each depth pixel is projected onto the surface; its sampling density is the photo pixels per
square metre there, (fx / depth)^2 (ignoring the angle to the surface). Every surface cell
(--cell metres) keeps the best density any photo reaches; a pixel's weight is its density over
that best, to the --power, never below --floor. Pixels without depth weigh 1.

Writes <out>/<photo stem>.png: weights 0..1 as 0..255 at the depth map's resolution
(train.py --pixel-weights upsamples them to the photo).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import gsdata


def pixels(im, depth, scale):
    """World points and densities of a photo's depth pixels."""
    h, w = depth.shape
    cam = im["cam"]
    fx, fy = cam["fx"] * scale[0], cam["fy"] * scale[1]
    cx, cy = cam["cx"] * scale[0], cam["cy"] * scale[1]
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0.05))
    z = depth[v, u]
    pc = np.stack([(u + .5 - cx) / fx * z, (v + .5 - cy) / fy * z, z], 1)
    R, t = im["viewmat"][:3, :3], im["viewmat"][:3, 3]
    return v, u, (pc - t) @ R, (cam["fx"] / z) ** 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--cell", type=float, default=0.003)
    p.add_argument("--floor", type=float, default=0.05)
    p.add_argument("--power", type=float, default=1.0)
    p.add_argument("--mode", choices=["relative", "absolute"], default="relative")
    p.add_argument("--min", type=float, default=0.5, help="absolute mode: lowest weight")
    p.add_argument("--max", type=float, default=2.0, help="absolute mode: highest weight")
    a = p.parse_args(argv)
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images, _, _ = gsdata.read_model(gsdata.find_model(root))
    if a.mode == "absolute":
        return absolute(a, root, out, images)
    per, best = [], {}
    for im in images:
        d, _ = gsdata.read_depth(root / "sensor_data", im["name"])
        if d is None:
            continue
        d = d.numpy()
        h, w = d.shape
        v, u, P, dens = pixels(im, d, (w / im["cam"]["w"], h / im["cam"]["h"]))
        keys = np.floor(P / a.cell).astype(np.int64)
        k = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
        per.append((im["name"], d.shape, v, u, k, dens))
        order = np.argsort(k)
        ks, ds = k[order], dens[order]
        starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
        for key, mx in zip(ks[starts], np.maximum.reduceat(ds, starts)):
            if mx > best.get(key, 0.0):
                best[key] = mx
    stats = []
    for name, shape, v, u, k, dens in per:
        wmap = np.ones(shape, np.float32)
        b = np.array([best[x] for x in k])
        wmap[v, u] = np.clip((dens / b) ** a.power, a.floor, 1.0)
        Image.fromarray((wmap * 255).round().astype(np.uint8)).save(out / (Path(name).stem + ".png"))
        stats.append((name, float(np.median(wmap[v, u])) if len(v) else 1.0))
    (out / "summary.json").write_text(json.dumps(dict(stats), indent=1))
    m = np.array([s for _, s in stats])
    print(f"{len(stats)} photos; median pixel weight per photo: median {np.median(m):.2f}, "
          f"p10 {np.percentile(m, 10):.2f}, p90 {np.percentile(m, 90):.2f}; {len(best):,} surface cells")


def patch_density(im, depth):
    """Photo pixels per square metre of surface for each depth pixel: (focal / depth)^2 times
    the cosine of the angle between the viewing ray and the surface normal (from neighbours)."""
    h, w = depth.shape
    cam = im["cam"]
    sx, sy = w / cam["w"], h / cam["h"]
    fx, fy, cx, cy = cam["fx"] * sx, cam["fy"] * sy, cam["cx"] * sx, cam["cy"] * sy
    u, v = np.meshgrid(np.arange(w) + .5, np.arange(h) + .5)
    ok = np.isfinite(depth) & (depth > 0.05)
    z = np.where(ok, depth, np.nan)
    P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)   # camera coordinates
    dx = np.full_like(P, np.nan); dy = np.full_like(P, np.nan)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    ray = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-12)
    cos = np.abs((n * ray).sum(-1))
    cos = np.where(np.isfinite(cos), np.clip(cos, 0.2, 1.0), 1.0)   # grazing angles capped at ~78 deg
    dens = (cam["fx"] / z) ** 2 * cos
    return np.where(ok, dens, np.nan)


def absolute(a, root, out, images):
    maps = []
    for im in images:
        d, _ = gsdata.read_depth(root / "sensor_data", im["name"])
        if d is None:
            continue
        maps.append((im["name"], patch_density(im, d.numpy())))
    # reference: the median photo's median patch (a pooled median would be set by close-ups,
    # which have far more patches inside the box than distant shots)
    med = float(np.median([np.nanmedian(m) for _, m in maps if np.isfinite(m).any()]))
    stats = {}
    for name, m in maps:
        wmap = np.where(np.isfinite(m), np.clip((m / med) ** a.power, a.min, a.max), 1.0).astype(np.float32)
        Image.fromarray((wmap / a.max * 255).round().astype(np.uint8)).save(out / (Path(name).stem + ".png"))
        stats[name] = float(np.median(wmap))
    (out / "summary.json").write_text(json.dumps({"scale": a.max, "median_density": med, "photos": stats}, indent=1))
    m = np.array(list(stats.values()))
    print(f"{len(stats)} photos; per-photo median weight {np.median(m):.2f} (p10 {np.percentile(m, 10):.2f}, "
          f"p90 {np.percentile(m, 90):.2f}); weights stored as value/{a.max:g}")


if __name__ == "__main__":
    main()
