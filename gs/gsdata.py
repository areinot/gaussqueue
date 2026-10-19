"""Shared I/O for the gsplat tools: COLMAP text models, SplatKing depth maps and
3DGS PLY files (the INRIA layout Brush and most viewers use).

Splats are kept as raw parameters, as stored in the PLY:
  means [N,3], quats [N,4] (wxyz, unnormalised), scales [N,3] (log),
  opacities [N] (logit), sh0 [N,1,3], shN [N,K,3] (K = (deg+1)^2 - 1).
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def keep_awake():
    """Ask Windows not to sleep while this process runs (no settings are changed;
    the request ends with the process). No-op elsewhere."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


# ---------------------------------------------------------------- PLY

def load_ply(path):
    with open(path, "rb") as f:
        header, props, n = [], [], 0
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                t, name = line.split()[1:3]
                if t not in ("float", "float32"):
                    raise ValueError(f"{path}: unsupported property type {t} for {name}")
                props.append(name)
            elif line == "end_header":
                break
        if "binary_little_endian" not in " ".join(header):
            raise ValueError(f"{path}: only binary little-endian PLY is supported")
        data = np.fromfile(f, dtype="<f4", count=n * len(props)).reshape(n, len(props))
    col = {p: i for i, p in enumerate(props)}

    def cols(names):
        return torch.from_numpy(np.ascontiguousarray(data[:, [col[c] for c in names]]))

    rest = sorted((p for p in props if p.startswith("f_rest_")), key=lambda p: int(p[7:]))
    k = len(rest) // 3
    splats = {
        "means": cols(["x", "y", "z"]),
        "quats": cols([f"rot_{i}" for i in range(4)]),
        "scales": cols([f"scale_{i}" for i in range(3)]),
        "opacities": cols(["opacity"])[:, 0],
        "sh0": cols([f"f_dc_{i}" for i in range(3)])[:, None, :],
        # f_rest is channel-major: all R coefficients, then G, then B
        "shN": (cols(rest).reshape(n, 3, k).transpose(1, 2) if k else torch.zeros(n, 0, 3)),
    }
    return splats


def save_ply(path, splats):
    s = {k: v.detach().float().cpu() for k, v in splats.items()}
    n, k = s["means"].shape[0], s["shN"].shape[1]
    names = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(3 * k)]
             + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = torch.cat([s["means"], s["sh0"].reshape(n, 3), s["shN"].transpose(1, 2).reshape(n, 3 * k),
                      s["opacities"][:, None], s["scales"], s["quats"]], dim=1).numpy().astype("<f4")
    head = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
            + "".join(f"property float {p}\n" for p in names) + "end_header\n")
    tmp = Path(str(path) + ".partial")
    with open(tmp, "wb") as f:
        f.write(head.encode("ascii"))
        data.tofile(f)
    os.replace(tmp, path)


def sh_degree_of(splats):
    return int(round(math.sqrt(splats["shN"].shape[1] + 1))) - 1


# ---------------------------------------------------------------- COLMAP

def _qvec_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_model(model_dir):
    """cameras.txt + images.txt (+ points3D.txt) from a COLMAP text model."""
    model_dir = Path(model_dir)
    cams = {}
    for line in (model_dir / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        cid, model, w, h, prm = int(p[0]), p[1], int(p[2]), int(p[3]), list(map(float, p[4:]))
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = prm[0]
            cx, cy = prm[1], prm[2]
        elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
            fx, fy, cx, cy = prm[:4]
        else:
            raise ValueError(f"unsupported camera model {model}")
        cams[cid] = dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
    images = []
    # Each image is a header line followed by its 2D-point line, which may be empty.
    lines = [l for l in (model_dir / "images.txt").read_text().splitlines() if not l.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    for line in lines[::2]:
        p = line.split()
        if len(p) < 10:
            continue
        q, t = np.array(list(map(float, p[1:5]))), np.array(list(map(float, p[5:8])))
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = _qvec_to_R(q / np.linalg.norm(q)), t
        images.append(dict(name=p[9], cam=cams[int(p[8])], viewmat=vm))
    images.sort(key=lambda im: im["name"])
    pts, rgb = read_points(model_dir / "points3D.txt")
    return images, pts, rgb


def read_points(path):
    """XYZ and RGB (0..1) from a COLMAP points3D.txt; empty arrays if it's missing."""
    pts = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                p = line.split()
                pts.append([float(v) for v in p[1:7]])
    pts = np.array(pts, dtype=np.float32).reshape(-1, 6)
    return pts[:, :3], pts[:, 3:6] / 255.0


def find_model(root):
    """The sparse model folder under a capture/trainable folder (sparse/0 or sparse)."""
    root = Path(root)
    for c in (root / "sparse" / "0", root / "sparse", root):
        if (c / "images.txt").exists():
            return c
    raise FileNotFoundError(f"no COLMAP text model under {root}")


# ---------------------------------------------------------------- dataset

def split(images, holdout_every):
    """Every Nth image (by name) is held out for evaluation; the rest train."""
    if not holdout_every:
        return images, []
    test = [im for i, im in enumerate(images) if i % holdout_every == holdout_every // 2]
    return [im for im in images if im not in test], test


class Views:
    """Images (uint8 on the GPU), intrinsics and view matrices at a capped resolution,
    plus LiDAR depth/confidence when a SplatKing sensor_data folder is given, and the
    per-photo masks of a box dataset (box.py: 255 train colour, 128 must be empty, 0 ignore)."""

    def __init__(self, root, images, max_res, device="cuda", depth_dir=None, weight_dir=None, photo_scales=None):
        """photo_scales: {photo name: factor <= 1}, an extra shrink for photos trained below the
        others' resolution (e.g. a blurred photo kept at the size where its blur is a normal edge)."""
        root = Path(root)
        photo_scales = photo_scales or {}
        self.names, self.rgb, self.K, self.viewmats, self.depth, self.conf = [], [], [], [], [], []
        self.masks, self.weights = [], []
        for im in images:
            img = Image.open(root / "images" / im["name"]).convert("RGB")
            cam = im["cam"]
            s = min(1.0, max_res / max(img.size)) if max_res else 1.0
            s *= min(1.0, float(photo_scales.get(im["name"], 1.0)))
            if s < 1.0:
                img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
            sx, sy = img.width / cam["w"], img.height / cam["h"]
            self.K.append(torch.tensor([[cam["fx"] * sx, 0, cam["cx"] * sx],
                                        [0, cam["fy"] * sy, cam["cy"] * sy], [0, 0, 1]], dtype=torch.float32))
            self.rgb.append(torch.from_numpy(np.array(img)).to(device))
            mp = root / "masks" / (Path(im["name"]).stem + ".png")
            m = None
            if mp.exists():
                mi = Image.open(mp)
                if mi.size != img.size:
                    mi = mi.resize(img.size, Image.NEAREST)
                m = torch.from_numpy(np.array(mi)).to(device)
            self.masks.append(m)
            wt = None
            wp = Path(weight_dir) / (Path(im["name"]).stem + ".png") if weight_dir else None
            if wp is not None and wp.exists():
                wi = Image.open(wp).resize(img.size, Image.BILINEAR)
                wt = (torch.from_numpy(np.array(wi)).float() / 255).half().to(device)
            self.weights.append(wt)
            self.viewmats.append(torch.tensor(im["viewmat"], dtype=torch.float32))
            self.names.append(im["name"])
            d = c = None
            if depth_dir:
                d, c = read_depth(depth_dir, im["name"])
            self.depth.append(None if d is None else d.to(device))
            self.conf.append(None if c is None else c.to(device))
        self.K = torch.stack(self.K).to(device) if self.K else None
        self.viewmats = torch.stack(self.viewmats).to(device) if self.viewmats else None
        self.has_depth = any(d is not None for d in self.depth)

    def __len__(self):
        return len(self.names)

    def image(self, i):
        return self.rgb[i].float() / 255.0

    def size(self, i):
        h, w = self.rgb[i].shape[:2]
        return w, h

    def centers(self):
        return torch.linalg.inv(self.viewmats)[:, :3, 3]


def read_depth(depth_dir, image_name):
    """SplatKing/ARKit depth for one photo: float32 metres and uint8 confidence (0 low,
    1 medium, 2 high), both 256x192 in the photo's sensor orientation."""
    stem = Path(image_name).stem
    dp, cp = Path(depth_dir) / f"{stem}_depth.bin", Path(depth_dir) / f"{stem}_confidence.bin"
    if not dp.exists():
        return None, None
    meta = Path(depth_dir) / f"{stem}.json"
    w, h = 256, 192
    if meta.exists():
        import json
        for aux in json.loads(meta.read_text()).get("auxiliaryOutputs", []):
            if aux.get("type") == "depth":
                w, h = int(aux["width"]), int(aux["height"])
    d = np.fromfile(dp, dtype="<f4")
    if d.size != w * h:
        return None, None
    d = torch.from_numpy(d.reshape(h, w).copy())
    c = (torch.from_numpy(np.fromfile(cp, dtype=np.uint8).reshape(h, w).copy())
         if cp.exists() else torch.full((h, w), 2, dtype=torch.uint8))
    return d, c


# ---------------------------------------------------------------- metrics

def _gauss_window(size=11, sigma=1.5, device="cuda"):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a, b, mask=None, weight=None):
    """SSIM of two [H,W,3] images in 0..1 (differentiable); with a [H,W] bool mask, the
    mean over the masked pixels only; with a [H,W] weight, the weighted mean."""
    import torch.nn.functional as F
    if a.device.type == "mps":   # Apple GPU: fused Metal SSIM (gs/mps_ssim.py)
        import mps_ssim
        g = _gauss_window(device=a.device)[0, 0].sum(0)
        return mps_ssim.ssim(a, b, g / g.sum(), mask, weight)
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    w = _gauss_window(device=a.device)
    if a.device.type == "mps":   # Apple GPU: the same blur as two 1D passes over all 15 channels at once
        g = w[0, 0].sum(0, keepdim=True)   # the window is separable: its row sums are the 1D taps
        g = g / g.sum()
        import mps_ssim
        blur = lambda t: mps_ssim.blur(t, g[0])
        mu_x, mu_y, exx, eyy, exy = blur(torch.cat([x, y, x * x, y * y, x * y], 1)).split(3, 1)
        sxx, syy, sxy = exx - mu_x ** 2, eyy - mu_y ** 2, exy - mu_x * mu_y
    else:
        mu_x, mu_y = F.conv2d(x, w, groups=3), F.conv2d(y, w, groups=3)
        sxx = F.conv2d(x * x, w, groups=3) - mu_x ** 2
        syy = F.conv2d(y * y, w, groups=3) - mu_y ** 2
        sxy = F.conv2d(x * y, w, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    k = (w.shape[-1] - 1) // 2
    if weight is not None:
        ww = weight[k:-k, k:-k].float() * (mask[k:-k, k:-k] if mask is not None else 1)
        return (m[0].mean(0) * ww).sum() / ww.sum().clamp_min(1e-6)
    if mask is None:
        return m.mean()
    mm = mask[k:-k, k:-k]
    return m[0].mean(0)[mm].mean() if mm.any() else m.mean()


def psnr(a, b, mask=None):
    e = (a - b) ** 2
    e = e[mask] if mask is not None and mask.any() else e
    return float(-10 * torch.log10(e.mean().clamp_min(1e-10)))


def save_png(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((img.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(path)
