"""Resample a capture's LiDAR depth maps into the undistorted photos' geometry.

  python undistort_depth.py --calibrated <cameras.txt + images.txt with RADIAL/OPENCV cameras>
                            --undistorted <the undistorted model: cameras.txt + images.txt (PINHOLE)>
                            --depth-dir <sensor_data> --out <new sensor_data>

When photos are undistorted for training (lens correction), their depth maps must follow, or
depth is looked up a few pixels off towards the edges. For every pixel of an output depth map
(same resolution as the input, covering the undistorted photo), the matching point in the
original, distorted photo is found through the calibrated lens and its depth (and confidence)
is taken from there (nearest sample, so depth edges stay sharp). The per-photo JSON files are
copied with the new size.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def cameras(path):
    out = {}
    for l in open(path):
        if l.strip() and not l.startswith("#"):
            p = l.split()
            out[p[0]] = (p[1], int(p[2]), int(p[3]), list(map(float, p[4:])))
    return out


def images(path):
    lines = [l for l in open(path) if not l.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    return {lines[k].split()[9]: lines[k].split()[8] for k in range(0, len(lines), 2) if len(lines[k].split()) >= 10}


def distort(model, prm, x, y):
    """Normalised undistorted coordinates -> normalised distorted ones."""
    if model in ("SIMPLE_RADIAL", "RADIAL"):
        k1, k2 = prm[3], (prm[4] if model == "RADIAL" else 0.0)
        r2 = x * x + y * y
        f = 1 + k1 * r2 + k2 * r2 * r2
        return x * f, y * f
    if model == "OPENCV":
        k1, k2, p1, p2 = prm[4:8]
        r2 = x * x + y * y
        f = 1 + k1 * r2 + k2 * r2 * r2
        return (x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * f + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    return x, y


def intrinsics(model, prm):
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return prm[0], prm[0], prm[1], prm[2]
    return prm[0], prm[1], prm[2], prm[3]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrated", required=True)
    p.add_argument("--undistorted", required=True)
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cal, und = Path(a.calibrated), Path(a.undistorted)
    ccam, cimg = cameras(cal / "cameras.txt"), images(cal / "images.txt")
    ucam, uimg = cameras(und / "cameras.txt"), images(und / "images.txt")
    src, out = Path(a.depth_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, ucid in uimg.items():
        stem = Path(name).stem
        meta = src / f"{stem}.json"
        dp, cp = src / f"{stem}_depth.bin", src / f"{stem}_confidence.bin"
        if not (meta.exists() and dp.exists() and name in cimg):
            continue
        j = json.loads(meta.read_text())
        aux = next((x for x in j.get("auxiliaryOutputs", []) if x.get("type") == "depth"), None)
        w, h = (int(aux["width"]), int(aux["height"])) if aux else (256, 192)
        depth = np.fromfile(dp, dtype="<f4").reshape(h, w)
        conf = np.fromfile(cp, dtype=np.uint8).reshape(h, w) if cp.exists() else None
        cm, cw, ch, cprm = ccam[cimg[name]]
        um, uw, uh, uprm = ucam[ucid]
        fx, fy, cx, cy = intrinsics(cm, cprm)
        ufx, ufy, ucx, ucy = intrinsics(um, uprm)
        # each output depth pixel's centre, in the undistorted photo's pixels
        u = (np.arange(w) + 0.5) * uw / w
        v = (np.arange(h) + 0.5) * uh / h
        U, V = np.meshgrid(u, v)
        x, y = (U - ucx) / ufx, (V - ucy) / ufy
        xd, yd = distort(cm, cprm, x, y)
        # -> the original photo's pixels -> the original depth map's samples
        ud, vd = xd * fx + cx, yd * fy + cy
        iu = np.floor(ud * w / cw).astype(int)
        iv = np.floor(vd * h / ch).astype(int)
        ok = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
        iu, iv = np.clip(iu, 0, w - 1), np.clip(iv, 0, h - 1)
        nd = np.where(ok, depth[iv, iu], 0).astype("<f4")
        nd.tofile(out / f"{stem}_depth.bin")
        if conf is not None:
            np.where(ok, conf[iv, iu], 0).astype(np.uint8).tofile(out / f"{stem}_confidence.bin")
        shutil.copyfile(meta, out / f"{stem}.json")
        n += 1
    print(f"{n} depth maps resampled into the undistorted geometry -> {out}")


if __name__ == "__main__":
    main()
