"""Cut a small box out of a capture, so experiments train in minutes instead of hours.

  python box.py <data> --out <dir> --box CX CY CZ SX SY SZ [--yaw DEG] [--frame meta.json]
                [--depth-dir <sensor_data>] [--min-pixels 400] [--points <points3D.txt> ...]

<data> holds images/ and a COLMAP text model (sparse/0), as for train.py. The box is given by
its centre and full size (metres), turned --yaw degrees about the vertical axis. With --frame,
those numbers are in the coordinates of a published result (its meta.json says how that result
was flipped and moved to its start view), e.g. read off a viewer's clipping box; without it,
they're in the capture's own (training) coordinates.

Writes a dataset train.py reads as it is:
  images/        each photo that sees the box, cropped to the box (full resolution)
  sparse/0/      one PINHOLE camera per crop (shifted principal point), same poses,
                 points3D.txt with only the points inside the box
  masks/         per crop, from the LiDAR depth: 255 = the surface seen there is inside the box
                 (train colour), 128 = the ray crosses the box and hits something behind it, so
                 the box must be empty there (train transparency), 0 = blocked by something in
                 front of the box, or no depth (ignored)
  sensor_data/   the depth maps cropped to match, confidence zeroed outside the box
  box.json       the box in training coordinates (train.py --box keeps splats inside it)

The crops are aligned to the depth map's pixel grid so the cropped depth lines up exactly.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import gsdata


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--box", type=float, nargs=6, required=True, metavar=("CX", "CY", "CZ", "SX", "SY", "SZ"))
    p.add_argument("--yaw", type=float, default=0.0, help="box rotation about the vertical axis, degrees")
    p.add_argument("--frame", help="meta.json of the result the box was measured in")
    p.add_argument("--viewer-unflips", action="store_true",
                   help="the viewer showed that (flipped) result the right way up, Y up, and its box "
                        "numbers are in those display coordinates (3D Splat App does this)")
    p.add_argument("--depth-dir", help="SplatKing sensor_data folder (default: found under <data>)")
    p.add_argument("--points", nargs="*", help="points3D.txt files to keep the inside points of "
                                               "(default: the model's own)")
    p.add_argument("--min-pixels", type=int, default=400,
                   help="skip photos with fewer depth pixels on surfaces inside the box")
    p.add_argument("--margin", type=int, default=1, help="crop margin, in depth pixels")
    return p.parse_args(argv)


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def box_in_training(center, size, yaw, meta=None, unflip=False):
    """(centre [3], axes [3,3] whose columns are the box's axes, half sizes [3]) in training
    coordinates. A published result is p_out = Ry(yaw_sv) (F p - origin), F = flip about X;
    a viewer that un-flips it shows F p_out."""
    c, A = np.array(center, float), rot_y(yaw)
    if unflip:
        F = np.diag([1.0, -1.0, -1.0])
        c, A = F @ c, F @ A
    if meta:
        F = np.diag([1.0, -1.0, -1.0]) if (meta.get("rotation_xyz_degrees") or [0])[0] == 180 else np.eye(3)
        sv = meta.get("start_view") or {}
        Rs, o = rot_y(sv.get("yaw_degrees", 0.0)), np.array(sv.get("origin", [0, 0, 0]), float)
        c = F @ (Rs.T @ c + o)
        A = F @ Rs.T @ A
    return c, A, np.array(size, float) / 2


def find_depth_dir(root):
    for d in [root / "sensor_data", *root.glob("*/sensor_data"), *root.parent.glob("**/sensor_data")]:
        if d.is_dir():
            return d
    return None


def classify(im, depth, conf, center, axes, half, scale):
    """Per depth pixel: 2 surface inside the box, 1 ray crosses the box but the surface is
    behind it, 0 otherwise (blocked in front, misses the box, or no depth)."""
    h, w = depth.shape
    cam = im["cam"]
    fx, fy = cam["fx"] * scale[0], cam["fy"] * scale[1]
    cx, cy = cam["cx"] * scale[0], cam["cy"] * scale[1]
    vm = im["viewmat"]
    R, t = vm[:3, :3], vm[:3, 3]
    C = -R.T @ t
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1) @ R   # world, per unit z
    # In box coordinates: p(z) = o + z * d; slab test for the z range inside the box.
    o = axes.T @ (C - center)
    d = rays @ axes
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - o) / d, (half - o) / d
    tmin = np.where(np.isfinite(t1), np.minimum(t1, t2), -np.inf)
    tmax = np.where(np.isfinite(t1), np.maximum(t1, t2), np.inf)
    # rays parallel to a slab: inside it for all z, or never
    par = ~np.isfinite(t1)
    outside_slab = par & (np.abs(o) > half)
    tmin[par], tmax[par] = -np.inf, np.inf
    enter = np.max(tmin, -1).clip(min=0)
    leave = np.min(tmax, -1)
    hits = (leave > enter) & ~outside_slab.any(-1)
    ok = np.isfinite(depth) & (depth > 0.05)
    lab = np.zeros((h, w), np.uint8)
    lab[hits & ok & (depth >= enter) & (depth <= leave)] = 2
    lab[hits & ok & (depth > leave)] = 1
    return lab


def erode(lab, n=1):
    """Drop labelled pixels next to ignored ones (the depth map is coarse: a 1920 px photo
    has 7.5 image pixels per depth pixel), so edges against occluders aren't trained."""
    keep = lab > 0
    for _ in range(n):
        k = keep.copy()
        k[1:] &= keep[:-1]
        k[:-1] &= keep[1:]
        k[:, 1:] &= keep[:, :-1]
        k[:, :-1] &= keep[:, 1:]
        keep = k
    return np.where(keep, lab, 0).astype(np.uint8)


def main(a):
    root, out = Path(a.data), Path(a.out)
    model = gsdata.find_model(root)
    images, pts, rgb = gsdata.read_model(model)
    depth_dir = Path(a.depth_dir) if a.depth_dir else find_depth_dir(root)
    if depth_dir is None:
        raise SystemExit("no sensor_data folder (LiDAR depth) found; pass --depth-dir")
    meta = json.loads(Path(a.frame).read_text()) if a.frame else None
    center, axes, half = box_in_training(a.box[:3], a.box[3:], a.yaw, meta, a.viewer_unflips)

    if out.exists():
        shutil.rmtree(out)
    for d in ("images", "sparse/0", "masks", "sensor_data"):
        (out / d).mkdir(parents=True)

    cams, img_lines, kept, pixels = [], [], [], 0
    for im in images:
        depth, conf = gsdata.read_depth(depth_dir, im["name"])
        if depth is None:
            continue
        depth, conf = depth.numpy(), conf.numpy()
        h, w = depth.shape
        W, H = im["cam"]["w"], im["cam"]["h"]
        lab = erode(classify(im, depth, conf, center, axes, half, (w / W, h / H)))
        inside = int((lab == 2).sum())
        if inside < a.min_pixels:
            continue
        ys, xs = np.nonzero(lab)
        y0, y1 = int(max(0, ys.min() - a.margin)), int(min(h, ys.max() + 1 + a.margin))
        x0, x1 = int(max(0, xs.min() - a.margin)), int(min(w, xs.max() + 1 + a.margin))
        # Crop the photo along the same depth-pixel boundaries (rounded to whole pixels).
        sx, sy = W / w, H / h
        X0, X1, Y0, Y1 = round(x0 * sx), round(x1 * sx), round(y0 * sy), round(y1 * sy)
        photo = Image.open(root / "images" / im["name"]).convert("RGB")
        if photo.size != (W, H):
            raise SystemExit(f"{im['name']} is {photo.size}, the model says {W}x{H}")
        stem = Path(im["name"]).stem
        photo.crop((X0, Y0, X1, Y1)).save(out / "images" / f"{stem}.png")
        mask = Image.fromarray(np.where(lab == 2, 255, np.where(lab == 1, 128, 0)).astype(np.uint8)[y0:y1, x0:x1])
        mask.resize((X1 - X0, Y1 - Y0), Image.NEAREST).save(out / "masks" / f"{stem}.png")
        dc = np.ascontiguousarray(depth[y0:y1, x0:x1], dtype="<f4")
        cc = np.where(lab == 2, conf, 0).astype(np.uint8)[y0:y1, x0:x1]
        dc.tofile(out / "sensor_data" / f"{stem}_depth.bin")
        np.ascontiguousarray(cc).tofile(out / "sensor_data" / f"{stem}_confidence.bin")
        (out / "sensor_data" / f"{stem}.json").write_text(json.dumps(
            {"auxiliaryOutputs": [{"type": "depth", "width": x1 - x0, "height": y1 - y0}]}))
        cam, cid = im["cam"], len(cams) + 1
        cams.append(f"{cid} PINHOLE {X1 - X0} {Y1 - Y0} {cam['fx']} {cam['fy']} "
                    f"{cam['cx'] - X0} {cam['cy'] - Y0}")
        R = im["viewmat"][:3, :3]
        q = rotmat_to_qvec(R)
        tv = im["viewmat"][:3, 3]
        img_lines += [f"{cid} {' '.join(f'{x:.9f}' for x in q)} {' '.join(f'{x:.9f}' for x in tv)} "
                      f"{cid} {stem}.png", ""]
        kept.append({"name": im["name"], "crop": [X0, Y0, X1 - X0, Y1 - Y0], "inside_px": inside})
        pixels += (X1 - X0) * (Y1 - Y0)

    if not kept:
        raise SystemExit("no photo sees a surface inside the box; check the box (and --frame)")
    (out / "sparse/0/cameras.txt").write_text("\n".join(cams) + "\n")
    (out / "sparse/0/images.txt").write_text("\n".join(img_lines) + "\n")

    sources = [Path(p) for p in a.points] if a.points else [model / "points3D.txt"]
    lines, n_in = [], 0
    for src in sources:
        if not src.exists():
            continue
        P, C = gsdata.read_points(src)
        if not len(P):
            continue
        local = (P - center) @ axes
        sel = (np.abs(local) <= half).all(-1)
        for p, c in zip(P[sel], C[sel]):
            n_in += 1
            lines.append(f"{n_in} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                         f"{int(round(c[0] * 255))} {int(round(c[1] * 255))} {int(round(c[2] * 255))} 0")
    (out / "sparse/0/points3D.txt").write_text("\n".join(lines) + "\n")

    info = {"center": center.tolist(), "axes": axes.tolist(), "half": half.tolist(),
            "given": {"box": a.box, "yaw": a.yaw, "frame": bool(meta)},
            "photos": len(kept), "photos_total": len(images), "points": n_in,
            "megapixels": round(pixels / 1e6, 1), "crops": kept}
    (out / "box.json").write_text(json.dumps(info, indent=1))
    print(f"{len(kept)}/{len(images)} photos see the box; {n_in:,} points inside; "
          f"{pixels / 1e6:.0f} MP of crops (the full photos are {len(images) * images[0]['cam']['w'] * images[0]['cam']['h'] / 1e6:.0f} MP)")


def rotmat_to_qvec(R):
    """COLMAP quaternion (w, x, y, z) of a rotation matrix."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


if __name__ == "__main__":
    main(parse())
