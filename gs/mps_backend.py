"""Apple GPU (Metal / MPS) backend for train.py, on metalsplat's kernels.

gsplat's rasterizer is CUDA-only. On a Mac, train.py renders through this module instead:
metalsplat (github.com/tchauffi/metalsplat, MIT) projects and alpha-composites the splats with
Metal kernels, and this wraps it in the subset of gsplat.rasterization() that train.py uses,
so the rest of the recipe (losses, schedule, MCMC/hybrid densification, carving, sweep
correction) is unchanged.

  - colour: SH evaluated with metalsplat's kernel, plus gsplat's +0.5 offset
  - depth and alpha: a second compositing pass whose "colour" is (camera depth, 1, 0), so the
    expected depth (gsplat's "ED") and the opacity are differentiable, as with gsplat; the
    LiDAR depth loss and the empty-space loss work unchanged
  - meta: means2d, conics, radii, opacities, gaussian_ids (the visible splats), width, height,
    n_cameras; absgrad accumulated by metalsplat's backward (AbsGS-style) for hybrid
  - gsplat's MCMC strategy is pure PyTorch apart from two CUDA helpers (covariances for the
    position noise, and MCMC's relocation formula); patch_gsplat_strategy() swaps in PyTorch
    versions, so gsplat (the Python package, no CUDA build) is needed only for its strategies.

Not available here: 2DGS mode, camera pose/sweep optimisation inside training (gradients don't
flow into the cameras), packed=False layouts. Needs PyTorch with MPS and torch.mps.compile_shader
(2.14+) and metalsplat on PYTHONPATH.
"""
import torch
import torch.nn.functional as F

NEAR = 0.01   # gsplat's default near plane; metalsplat's 0.2 m would cut close-up captures
AUX_SCALE = 0.5   # depth/opacity pass resolution (the LiDAR loss compares at ~256x192 anyway)
_K_CACHE = {}


def _intrinsics(K):
    """fx, fy, cx, cy as floats, read back from the GPU once per camera (each read would make
    the CPU wait for the GPU, every step)."""
    key = (K.data_ptr(), K.storage_offset())
    if key not in _K_CACHE:
        k = K.detach().cpu()
        _K_CACHE[key] = (float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))
    return _K_CACHE[key]


def available():
    if not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")):
        return False
    try:
        import metalsplat  # noqa: F401
    except ImportError:
        return False
    return True


def quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=True, triu=False):
    """PyTorch stand-in for gsplat's CUDA helper (only what MCMC's noise injection uses)."""
    w, x, y, z = F.normalize(quats, dim=-1).unbind(-1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)
    M = R * scales[:, None, :]
    covar = M @ M.transpose(1, 2) if compute_covar else None
    preci = None
    if compute_preci:
        Mi = R / scales[:, None, :]
        preci = Mi @ Mi.transpose(1, 2)
    if triu:
        idx = torch.triu_indices(3, 3)
        covar = covar[:, idx[0], idx[1]] if covar is not None else None
        preci = preci[:, idx[0], idx[1]] if preci is not None else None
    return covar, preci


def compute_relocation(opacities, scales, ratios, binoms):
    """PyTorch version of gsplat's relocation kernel (MCMC paper, eq. 9): a splat copied into
    `ratio` splats gets opacity 1 - (1 - o)^(1/ratio) and its scale shrunk so the sum of the
    copies matches the original."""
    n_max = binoms.shape[0]
    ratios = ratios.clamp(1, n_max).long()
    new_o = 1 - (1 - opacities) ** (1.0 / ratios.to(opacities.dtype))
    denom = torch.zeros_like(opacities)
    k = torch.arange(n_max, device=opacities.device, dtype=opacities.dtype)
    sign_sqrt = ((-1.0) ** k) / torch.sqrt(k + 1)                       # [K]
    powers = new_o[:, None] ** (k[None, :] + 1)                          # [N, K]
    for i in range(1, n_max + 1):
        use = (ratios >= i)
        if not use.any():
            break
        terms = binoms[i - 1].to(opacities.dtype)[None, :i] * sign_sqrt[None, :i] * powers[:, :i]
        denom = denom + torch.where(use, terms.sum(-1), torch.zeros_like(denom))
    new_s = (opacities / denom)[:, None] * scales
    return new_o, new_s


def rotmat_to_quat(R):
    """(w, x, y, z) of a 3x3 rotation, differentiable (Shepperd's method, largest-component branch)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    cands = torch.stack([
        torch.stack([1 + tr, m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]),
        torch.stack([m[2, 1] - m[1, 2], 1 + m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0]]),
        torch.stack([m[0, 2] - m[2, 0], m[0, 1] + m[1, 0], 1 - m[0, 0] + m[1, 1] - m[2, 2], m[1, 2] + m[2, 1]]),
        torch.stack([m[1, 0] - m[0, 1], m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], 1 - m[0, 0] - m[1, 1] + m[2, 2]])])
    k = int(torch.stack([1 + tr, 1 + m[0, 0] - m[1, 1] - m[2, 2], 1 - m[0, 0] + m[1, 1] - m[2, 2],
                         1 - m[0, 0] - m[1, 1] + m[2, 2]]).detach().argmax())
    q = cands[k]
    return q / q.norm()


def quat_mul(a, b):
    """Hamilton product, (w, x, y, z); a [4] or [N, 4], b [N, 4]."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def patch_gsplat_strategy():
    """Point gsplat's strategy helpers at the PyTorch versions above."""
    import gsplat.strategy.ops as ops
    ops.quat_scale_to_covar_preci = quat_scale_to_covar_preci
    ops.compute_relocation = compute_relocation


def rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                  sh_degree=None, render_mode="RGB+ED", packed=True, absgrad=False,
                  rasterize_mode="classic", **_):
    """gsplat.rasterization() for one camera on the Apple GPU. Returns (render [1,H,W,4]: RGB +
    expected depth, alpha [1,H,W,1], meta)."""
    from metalsplat.ops.project import project_gaussians
    from metalsplat.ops.rasterize import rasterize_gaussians
    from metalsplat.ops.sh import eval_sh
    assert viewmats.shape[0] == 1, "one camera per call"
    vm = viewmats[0]
    fx, fy, cx, cy = _intrinsics(Ks[0])
    if vm.requires_grad:   # metalsplat gives the camera no gradient: move the scene into its frame instead
        R, t = vm[:3, :3], vm[:3, 3]
        eye = torch.eye(3, device=means.device)
        args3d = ((means @ R.T + t).contiguous(), scales.contiguous(),
                  quat_mul(rotmat_to_quat(R), F.normalize(quats, dim=-1)).contiguous(), eye, torch.zeros(3, device=means.device))
    else:
        args3d = (means.contiguous(), scales.contiguous(), F.normalize(quats, dim=-1).contiguous(),
                  vm[:3, :3].contiguous(), vm[:3, 3].contiguous())
    m2d, depths, conics, radii, valid, comp = project_gaussians(
        *args3d, fx, fy, cx, cy, int(width), int(height), near=NEAR, eps2d=0.3)
    opac = opacities * comp if rasterize_mode == "antialiased" else opacities
    if sh_degree is None:   # colours given directly
        rgb_in = colors
    else:
        cam = -vm[:3, :3].T @ vm[:3, 3]
        dirs = F.normalize(means - cam, dim=-1)
        rgb_in = (eval_sh(colors.contiguous(), dirs.contiguous(), int(sh_degree)) + 0.5).clamp_min(0)
    n = len(means)
    accum = torch.zeros(n, device=means.device) if absgrad else None
    image = rasterize_gaussians(m2d, depths, conics, opac, rgb_in.contiguous(), radii, valid,
                                int(width), int(height), abs_grad_accum=accum)
    # second pass, at AUX_SCALE: (depth, 1, 0) as colour -> differentiable sum(w z) and sum(w)
    sw, sh_ = max(1, round(width * AUX_SCALE)), max(1, round(height * AUX_SCALE))
    f = sw / width
    m2s, dps, cos, ras, vas, cps = project_gaussians(
        *args3d, fx * f, fy * f, cx * f, cy * f, sw, sh_, near=NEAR, eps2d=0.3)
    ops = opacities * cps if rasterize_mode == "antialiased" else opacities
    z = (means @ vm[:3, :3].T + vm[:3, 3])[:, 2:3]
    dz = torch.cat([z, torch.ones_like(z), torch.zeros_like(z)], -1).contiguous()
    dz_img = rasterize_gaussians(m2s, dps, cos, ops, dz, ras, vas, sw, sh_)
    if (sw, sh_) != (int(width), int(height)):
        dz_img = F.interpolate(dz_img.permute(2, 0, 1)[None], size=(int(height), int(width)),
                               mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    alpha = dz_img[..., 1:2]
    ed = dz_img[..., 0:1] / alpha.clamp_min(1e-6)
    out = torch.cat([image, ed], -1)[None]
    vis = (valid > 0) & (radii > 0)
    ids = vis.nonzero(as_tuple=True)[0]
    meta = {"means2d": m2d, "conics": conics[ids], "radii": radii[ids].float()[:, None].expand(-1, 2),
            "opacities": opac[ids], "gaussian_ids": ids, "width": int(width), "height": int(height),
            "n_cameras": 1, "absgrad_full": accum, "radii_full": radii}
    return out, alpha[None], meta
