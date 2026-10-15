"""Per-pixel training weights from surface coverage: where close-up pixels sample a surface
densely, distant pixels of the same surface step back instead of being averaged in.

  python coverage.py <data> --out <dir> [--cell 0.003] [--floor 0.05] [--power 1]

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
    a = p.parse_args(argv)
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images, _, _ = gsdata.read_model(gsdata.find_model(root))
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


if __name__ == "__main__":
    main()
