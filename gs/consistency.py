"""How much do a capture's camera poses disagree about a region? No trained model needed.

  python consistency.py <model dir with images.txt/cameras.txt> --images <photos> --depth-dir <sensor_data>
                        --box CX CY CZ SX SY SZ [--yaw DEG] [--frame meta.json] [--viewer-unflips]
                        [--neighbours 8] [--out report.json]

For each photo, the pixels whose LiDAR surface lies inside the box are lifted to 3D with that
photo's own depth and pose, projected into each of its nearest neighbouring photos, and the
neighbour's colours pulled back: the neighbour warped into this photo's view. If the two poses
agree, the warped region lines up with the photo; the offset between them (phase correlation
on image edges) is how far the poses disagree there, in this photo's pixels. A photo's score is
its median offset over its neighbours.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import box as boxmod
import gsdata

W, H = 640, 480   # working resolution (photos are 1920x1440 or close)


def edges(x):
    gy, gx = np.gradient(x)
    return np.hypot(gx, gy)


def offset(a, b, m):
    """Translation (px) of b relative to a over the box mask's bounding box."""
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
    i = np.unravel_index(np.argmax(c), c.shape)
    d = [(k if k <= n // 2 else k - n) for k, n in zip(i, c.shape)]
    return float(np.hypot(*d))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--images", required=True)
    p.add_argument("--depth-dir", required=True)
    p.add_argument("--box", type=float, nargs=6, required=True)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--frame")
    p.add_argument("--viewer-unflips", action="store_true")
    p.add_argument("--neighbours", type=int, default=8)
    p.add_argument("--out")
    a = p.parse_args(argv)
    meta = json.loads(Path(a.frame).read_text()) if a.frame else None
    center, axes, half = boxmod.box_in_training(a.box[:3], a.box[3:], a.yaw, meta, a.viewer_unflips)
    ims = gsdata.read_model(Path(a.model))[0]
    views = {}
    for im in ims:
        d, conf = gsdata.read_depth(a.depth_dir, im["name"])
        if d is None:
            continue
        cam = im["cam"]
        g = np.asarray(Image.open(Path(a.images) / im["name"]).convert("L").resize((W, H), Image.BILINEAR), float) / 255
        dd = np.asarray(Image.fromarray(d.numpy()).resize((W, H), Image.NEAREST))
        sx, sy = W / cam["w"], H / cam["h"]
        K = (cam["fx"] * sx, cam["fy"] * sy, cam["cx"] * sx, cam["cy"] * sy)
        R, t = im["viewmat"][:3, :3], im["viewmat"][:3, 3]
        v, u = np.mgrid[0:H, 0:W]
        z = dd
        Pc = np.stack([(u + .5 - K[2]) / K[0] * z, (v + .5 - K[3]) / K[1] * z, z], -1)
        Pw = (Pc - t) @ R
        inside = (np.abs((Pw - center) @ axes) <= half).all(-1) & (z > 0.05)
        views[im["name"]] = dict(g=g, K=K, R=R, t=t, Pw=Pw, inside=inside, C=-R.T @ t, fwd=R[2])
    names = sorted(views)
    rows = {}
    for n in names:
        V = views[n]
        if V["inside"].sum() < 400:
            continue
        cand = [m for m in names if m != n and views[m]["inside"].sum() > 400]
        cand.sort(key=lambda m: np.linalg.norm(views[m]["C"] - V["C"]) + 0.2 * (1 - V["fwd"] @ views[m]["fwd"]))
        offs = []
        for m in cand[:a.neighbours]:
            O = views[m]
            q = V["Pw"] @ O["R"].T + O["t"]
            ok = V["inside"] & (q[..., 2] > 0.05)
            uu = O["K"][0] * q[..., 0] / np.maximum(q[..., 2], 1e-6) + O["K"][2] - .5
            vv = O["K"][1] * q[..., 1] / np.maximum(q[..., 2], 1e-6) + O["K"][3] - .5
            ok &= (uu >= 0) & (uu < W - 1) & (vv >= 0) & (vv < H - 1)
            if ok.sum() < 400:
                continue
            ui, vi = np.clip(uu.astype(int), 0, W - 2), np.clip(vv.astype(int), 0, H - 2)
            fu, fv = np.clip(uu - ui, 0, 1), np.clip(vv - vi, 0, 1)
            g = O["g"]
            warp = (g[vi, ui] * (1 - fu) * (1 - fv) + g[vi, ui + 1] * fu * (1 - fv)
                    + g[vi + 1, ui] * (1 - fu) * fv + g[vi + 1, ui + 1] * fu * fv)
            warp = np.where(ok, warp, V["g"])
            o = offset(V["g"], warp, ok)
            if o is not None:
                offs.append(o * (1920 / W))   # in full-photo pixels
        if offs:
            rows[n] = float(np.median(offs))
    s = np.array(list(rows.values()))
    report = {"photos": len(s), "median_px": float(np.median(s)), "p90_px": float(np.percentile(s, 90)),
              "over_10px": int((s > 10).sum()), "over_20px": int((s > 20).sum()), "per_photo": rows}
    print(f"{len(s)} photos: pose disagreement on the region, median {report['median_px']:.1f} px, "
          f"p90 {report['p90_px']:.1f} px, >10 px: {report['over_10px']}, >20 px: {report['over_20px']}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
