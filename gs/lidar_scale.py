"""How far the capture's LiDAR depth is off from its photos' scale: one factor for train.py --depth-scale.

  python lidar_scale.py --model <sparse/0> --database <matches.db> --depth-dir <sensor_data> --out scale.json

Each verified feature match (COLMAP two_view_geometries) is triangulated with the model's poses and kept
when both reprojections are under 1.5 px and the two rays cross at more than 3 degrees. Where a photo's
own high-confidence LiDAR (away from depth steps) covers that point, LiDAR depth is compared with the
point's depth in that camera. iPhone LiDAR reads a few percent long against ARKit's (metric) poses; the
median ratio is written as "scale" = photo depth / LiDAR depth, with the error before and after on a
held-out half. "ok" is false (and train.py should keep its default) with too few comparisons.
"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

import gsdata

PAIR = 2147483647


def lidar_map(depth_dir, stem):
    """(depth with unusable pixels zeroed, fx, fy, cx, cy, sensor-pixels per depth pixel), or None."""
    meta_f = Path(depth_dir) / f"{stem}.json"
    meta = json.loads(meta_f.read_text()).get("metadata", {}).get("camera") if meta_f.exists() else None
    d, c = gsdata.read_depth(depth_dir, stem)
    if meta is None or d is None or "intrinsics" not in meta:
        return None
    d, c = d.numpy().astype(np.float64), c.numpy()
    ok = (c == 2) & (d > 0.05)
    centre = d[1:-1, 1:-1]
    ok[1:-1, 1:-1] &= np.all([np.abs(nb - centre) < 0.01 for nb in (d[1:-1, 2:], d[1:-1, :-2], d[2:, 1:-1], d[:-2, 1:-1])], 0)
    k = meta["intrinsics"]
    return np.where(ok, d, 0), k[0], k[4], k[6], k[7], meta["imageResolution"]["width"] / d.shape[1]


def comparisons(model, database, depth_dir, min_angle=3.0, max_px=1.5):
    """[n, 2] (photo depth, LiDAR depth) in metres."""
    ims = gsdata.read_model(Path(model))[0]
    by = {Path(im["name"]).stem: k for k, im in enumerate(ims)}
    c2w = [np.linalg.inv(im["viewmat"]) for im in ims]
    con = sqlite3.connect(database)
    iname = {i: Path(n).stem for i, n in con.execute("select image_id, name from images")}
    kps = {iname[i]: np.frombuffer(d, np.float32).reshape(r, c)[:, :2]
           for i, r, c, d in con.execute("select image_id, rows, cols, data from keypoints") if d}
    maps, out = {}, []

    def ray(k, x):
        cam = ims[k]["cam"]
        d = np.stack([(x[:, 0] - cam["cx"]) / cam["fx"], (x[:, 1] - cam["cy"]) / cam["fy"], np.ones(len(x))], 1) @ c2w[k][:3, :3].T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    for pid, rows, _, data in con.execute("select pair_id, rows, cols, data from two_view_geometries where rows > 0"):
        b = pid % PAIR
        a = (pid - b) // PAIR
        na, nb = iname.get(a), iname.get(b)
        if na not in by or nb not in by or na not in kps or nb not in kps:
            continue
        ka, kb = by[na], by[nb]
        m = np.frombuffer(data, np.uint32).reshape(rows, 2)
        xa, xb = kps[na][m[:, 0]], kps[nb][m[:, 1]]
        da, db = ray(ka, xa), ray(kb, xb)
        Ca, Cb = c2w[ka][:3, 3], c2w[kb][:3, 3]
        w0 = Ca - Cb
        bb = (da * db).sum(1)
        den = np.maximum(1 - bb * bb, 1e-9)
        s1 = (bb * (db @ w0) - da @ w0) / den
        s2 = ((db @ w0) - bb * (da @ w0)) / den
        X = 0.5 * (Ca + s1[:, None] * da + Cb + s2[:, None] * db)
        good = (s1 > 0.05) & (s2 > 0.05) & (np.degrees(np.arccos(np.clip(bb, -1, 1))) > min_angle)
        for k, x in ((ka, xa), (kb, xb)):
            cam, vm = ims[k]["cam"], ims[k]["viewmat"]
            pc = X @ vm[:3, :3].T + vm[:3, 3]
            good &= np.hypot(cam["fx"] * pc[:, 0] / pc[:, 2] + cam["cx"] - x[:, 0],
                             cam["fy"] * pc[:, 1] / pc[:, 2] + cam["cy"] - x[:, 1]) < max_px
        for k, stem in ((ka, na), (kb, nb)):
            if stem not in maps:
                maps[stem] = lidar_map(depth_dir, stem)
            L = maps[stem]
            if L is None or not good.any():
                continue
            D, fx, fy, cx, cy, sc = L
            vm = ims[k]["viewmat"]
            pc = X[good] @ vm[:3, :3].T + vm[:3, 3]
            z = pc[:, 2]
            u = np.round((fx * pc[:, 0] / z + cx + 0.5) / sc - 0.5).astype(int)
            v = np.round((fy * pc[:, 1] / z + cy + 0.5) / sc - 0.5).astype(int)
            inside = (u >= 0) & (u < D.shape[1]) & (v >= 0) & (v < D.shape[0])
            dl = np.zeros(len(z))
            dl[inside] = D[v[inside], u[inside]]
            sel = dl > 0
            out.append(np.stack([z[sel], dl[sel]], 1))
    out = np.concatenate(out) if out else np.zeros((0, 2))
    return out[np.abs(out[:, 1] - out[:, 0]) < 0.15]   # drop gross mismatches (occlusion, the wrong pixel)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--database", required=True)
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-comparisons", type=int, default=2000)
    a = p.parse_args(argv)
    A = comparisons(a.model, a.database, a.depth_dir)
    res = {"comparisons": int(len(A)), "ok": bool(len(A) >= a.min_comparisons)}
    if len(A):
        z, dl = A[:, 0], A[:, 1]
        half = np.random.default_rng(0).random(len(z)) < 0.5
        ratio = float(np.median(dl[half] / z[half])) if half.any() else float(np.median(dl / z))
        err = lambda pred: float(np.median(np.abs(dl - pred)[~half]) * 1000) if (~half).any() else None
        res.update(scale=round(1 / ratio, 5), lidar_longer_share=round(float(np.mean(dl > z)), 3),
                   error_mm_before=err(z), error_mm_after=err(ratio * z),
                   by_depth={f"{lo:.1f}-{hi:.1f} m": round(float(np.median(dl[m] / z[m])), 4)
                             for lo, hi in ((0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.5))
                             for m in [(z >= lo) & (z < hi)] if m.sum() >= 50})
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
