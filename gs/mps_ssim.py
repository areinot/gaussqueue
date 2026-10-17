"""SSIM's Gaussian blur as a Metal kernel, for training on the Apple GPU.

PyTorch's grouped convolutions are slow on MPS (SSIM took ~90 of ~135 ms per training step);
the 11-tap Gaussian is separable, so here it runs as two 1D passes of a small kernel compiled
with torch.mps.compile_shader, forward and backward (the blur is symmetric, so its backward is
the same blur over the zero-padded gradient). Same numbers as the convolution, to rounding.
"""
import torch

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// out[c, y, x] = sum_k taps[k] * in[c, y, x + k]   (horizontal, "valid": out width = W - K + 1)
// or            sum_k taps[k] * in[c, y + k, x]   (vertical)
kernel void blur_valid(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                       constant float* taps [[buffer(2)]], constant int& K [[buffer(3)]],
                       constant int& H [[buffer(4)]], constant int& W [[buffer(5)]],
                       constant int& vertical [[buffer(6)]], uint3 gid [[thread_position_in_grid]]) {
    int Wo = vertical ? W : W - K + 1, Ho = vertical ? H - K + 1 : H;
    int x = gid.x, y = gid.y, c = gid.z;
    if (x >= Wo || y >= Ho) return;
    device const float* s = src + (long)c * H * W;
    float acc = 0.0;
    if (vertical) { for (int k = 0; k < K; ++k) acc += taps[k] * s[(long)(y + k) * W + x]; }
    else          { for (int k = 0; k < K; ++k) acc += taps[k] * s[(long)y * W + x + k]; }
    dst[(long)c * Ho * Wo + (long)y * Wo + x] = acc;
}

// Backward of blur_valid: din[c, y, x] = sum_k taps[k] * dout[c, y, x - k] (in range),
// where (H, W) is the input's size.
kernel void blur_full(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                      constant float* taps [[buffer(2)]], constant int& K [[buffer(3)]],
                      constant int& H [[buffer(4)]], constant int& W [[buffer(5)]],
                      constant int& vertical [[buffer(6)]], uint3 gid [[thread_position_in_grid]]) {
    int Wo = vertical ? W : W - K + 1, Ho = vertical ? H - K + 1 : H;   // dout's size
    int x = gid.x, y = gid.y, c = gid.z;
    if (x >= W || y >= H) return;
    device const float* s = src + (long)c * Ho * Wo;
    float acc = 0.0;
    for (int k = 0; k < K; ++k) {
        int xs = vertical ? x : x - k, ys = vertical ? y - k : y;
        if (xs >= 0 && xs < Wo && ys >= 0 && ys < Ho) acc += taps[k] * s[(long)ys * Wo + xs];
    }
    dst[(long)c * H * W + (long)y * W + x] = acc;
}

// Fused SSIM map: from the 15 blurred statistics (mu_x, mu_y, E[xx], E[yy], E[xy], 3 channels
// each, layout [15, H, W]) and a per-pixel weight, writes the weighted map value and the
// gradient of sum(w * map) with respect to each statistic, in one pass.
kernel void ssim_map(device const float* B [[buffer(0)]], device const float* wt [[buffer(1)]],
                     device float* val [[buffer(2)]], device float* dB [[buffer(3)]],
                     constant int& HW [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= HW) return;
    float w = wt[gid], total = 0.0;
    const float c1 = 0.0001, c2 = 0.0009;
    for (int c = 0; c < 3; ++c) {
        float mx = B[(0 + c) * HW + gid], my = B[(3 + c) * HW + gid];
        float exx = B[(6 + c) * HW + gid], eyy = B[(9 + c) * HW + gid], exy = B[(12 + c) * HW + gid];
        float sxx = exx - mx * mx, syy = eyy - my * my, sxy = exy - mx * my;
        float A1 = 2 * mx * my + c1, A2 = 2 * sxy + c2;
        float B1 = mx * mx + my * my + c1, B2 = sxx + syy + c2;
        float m = (A1 * A2) / (B1 * B2);
        total += m;
        // d m / d (A1, A2, B1, B2)
        float dA1 = A2 / (B1 * B2), dA2 = A1 / (B1 * B2);
        float dB1 = -m / B1, dB2 = -m / B2;
        float s = w / 3.0;   // the map is averaged over the 3 channels
        // A1 = 2 mx my; A2 = 2 (exy - mx my); B1 = mx^2 + my^2; B2 = exx - mx^2 + eyy - my^2
        dB[(0 + c) * HW + gid] = s * (dA1 * 2 * my - dA2 * 2 * my + dB1 * 2 * mx - dB2 * 2 * mx);
        dB[(3 + c) * HW + gid] = s * (dA1 * 2 * mx - dA2 * 2 * mx + dB1 * 2 * my - dB2 * 2 * my);
        dB[(6 + c) * HW + gid] = s * dB2;
        dB[(9 + c) * HW + gid] = s * dB2;
        dB[(12 + c) * HW + gid] = s * dA2 * 2;
    }
    val[gid] = w * total / 3.0;
}
"""
_lib = None


def _kernels():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SRC)
    return _lib


def _pass(fn, t, taps, vertical, out_shape):
    C, H, W = t.shape
    out = torch.empty(out_shape, device=t.device, dtype=torch.float32)
    getattr(_kernels(), fn)(t.contiguous(), out, taps, len(taps), H, W, int(vertical),
                            threads=(out_shape[2], out_shape[1], C))
    return out


def _pass_full(t_out, taps, vertical, in_shape):
    """Backward of one valid pass: zero-pad the gradient by K-1 on both sides of that axis
    and blur it validly again (the taps are symmetric). Faster than a bounds-checked kernel."""
    K = len(taps)
    pad = (0, 0, K - 1, K - 1) if vertical else (K - 1, K - 1, 0, 0)
    padded = torch.nn.functional.pad(t_out, pad)
    return _pass("blur_valid", padded, taps, vertical, in_shape)


class _Blur(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, taps):   # t: [C, H, W] -> [C, H-K+1, W-K+1]
        C, H, W = t.shape
        K = len(taps)
        mid = _pass("blur_valid", t, taps, False, (C, H, W - K + 1))
        ctx.save_for_backward(taps)
        ctx.shapes = ((C, H, W), (C, H, W - K + 1))
        return _pass("blur_valid", mid, taps, True, (C, H - K + 1, W - K + 1))

    @staticmethod
    def backward(ctx, g):
        (taps,) = ctx.saved_tensors
        s_in, s_mid = ctx.shapes
        g_mid = _pass_full(g, taps, True, s_mid)
        return _pass_full(g_mid, taps, False, s_in), None


def blur(t, taps):
    """Valid separable blur of [1, C, H, W] (or [C, H, W]) with the 1D taps tensor (on MPS)."""
    squeeze = t.dim() == 4
    out = _Blur.apply(t[0] if squeeze else t, taps.float().contiguous())
    return out[None] if squeeze else out


class _SSIM(torch.autograd.Function):
    """sum(weight * ssim_map) for [3, H, W] images x, y; weight is [H-K+1, W-K+1], already
    normalised (sums to 1 for a mean). Only x gets a gradient."""

    @staticmethod
    def forward(ctx, x, y, weight, taps):
        stats = torch.cat([x, y, x * x, y * y, x * y], 0)          # [15, H, W]
        B = _Blur.forward(ctx, stats, taps)                          # [15, H', W'] (saves taps, shapes)
        HW = B.shape[1] * B.shape[2]
        val = torch.empty(HW, device=x.device)
        dB = torch.empty_like(B)
        _kernels().ssim_map(B, weight.contiguous().float(), val, dB, HW, threads=HW)
        ctx.save_for_backward(taps, x, y, dB)
        return val.sum()

    @staticmethod
    def backward(ctx, g):
        taps, x, y, dB = ctx.saved_tensors
        s_in, s_mid = ctx.shapes
        G = _pass_full(_pass_full(dB, taps, True, s_mid), taps, False, s_in) * g   # d/d stats
        dx = G[0:3] + 2 * x * G[6:9] + y * G[12:15]
        return dx, None, None, None


def ssim(a, b, taps, mask=None, weight=None):
    """gsdata.ssim on the Apple GPU: [H, W, 3] images, fused. Same mask/weight semantics."""
    x, y = a.permute(2, 0, 1), b.permute(2, 0, 1).detach()
    K = len(taps)
    k = (K - 1) // 2
    Hp, Wp = x.shape[1] - K + 1, x.shape[2] - K + 1
    if weight is not None:
        w = weight[k:-k, k:-k].float() * (mask[k:-k, k:-k] if mask is not None else 1)
    elif mask is not None and mask[k:-k, k:-k].any():
        w = mask[k:-k, k:-k].float()
    else:
        w = torch.ones(Hp, Wp, device=a.device)
    w = w / w.sum().clamp_min(1e-6)
    return _SSIM.apply(x.contiguous(), y.contiguous(), w, taps.float().contiguous())
