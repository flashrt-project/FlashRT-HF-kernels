"""FlashRT spatiotemporal layout kernels."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_ncdhw(x: torch.Tensor, name: str) -> None:
    if x.dim() != 5:
        raise RuntimeError(f"{name} must have shape (B, C, T, H, W)")


@torch.library.register_fake(add_op_namespace_prefix("ncdhw_to_blc_bf16"))
def _ncdhw_to_blc_bf16_fake(x: torch.Tensor, out: torch.Tensor) -> None:
    _check_ncdhw(x, "x")
    b, c, t, h, w = x.shape
    if out.shape != (b, t * h * w, c):
        raise RuntimeError("out must have shape (B, T * H * W, C)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("patch_im2col_bf16"))
def _patch_im2col_bf16_fake(x: torch.Tensor, out: torch.Tensor) -> None:
    if x.dim() != 4 or x.shape[1:] != (224, 224, 3):
        raise RuntimeError("x must have shape (num_views, 224, 224, 3)")
    if out.shape != (x.shape[0] * 256, 588):
        raise RuntimeError("out must have shape (num_views * 256, 588)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("time_unshuffle2_bf16"))
def _time_unshuffle2_bf16_fake(x: torch.Tensor, out: torch.Tensor) -> None:
    _check_ncdhw(x, "x")
    b, c2, t, h, w = x.shape
    if c2 % 2 != 0:
        raise RuntimeError("x.shape[1] must be even")
    if out.shape != (b, c2 // 2, 2 * t, h, w):
        raise RuntimeError("out must have shape (B, C / 2, 2 * T, H, W)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("add_bias_ncdhw_bf16"))
def _add_bias_ncdhw_bf16_fake(x: torch.Tensor, bias: torch.Tensor) -> None:
    _check_ncdhw(x, "x")
    if bias.shape != (x.shape[1],):
        raise RuntimeError("bias must have shape (C,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("update_cache2_ncdhw_bf16"))
def _update_cache2_ncdhw_bf16_fake(cur: torch.Tensor, prev: torch.Tensor, out: torch.Tensor) -> None:
    _check_ncdhw(cur, "cur")
    b, c, _, h, w = cur.shape
    if prev.shape != (b, c, 2, h, w) or out.shape != (b, c, 2, h, w):
        raise RuntimeError("prev and out must have shape (B, C, 2, H, W)")
    return None


def ncdhw_to_blc_bf16(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Convert BF16 NCDHW tensor to BLC where L = T * H * W."""

    if out is None:
        out = torch.empty((x.shape[0], x.shape[2] * x.shape[3] * x.shape[4], x.shape[1]), device=x.device, dtype=x.dtype)
    ops.ncdhw_to_blc_bf16(x, out)
    return out


def patch_im2col_bf16(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Convert BF16 NHWC images into flattened 14x14 patch rows."""

    if out is None:
        out = torch.empty((x.shape[0] * 256, 588), device=x.device, dtype=x.dtype)
    ops.patch_im2col_bf16(x, out)
    return out


def time_unshuffle2_bf16(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Convert BF16 (B, 2C, T, H, W) into (B, C, 2T, H, W)."""

    if out is None:
        out = torch.empty((x.shape[0], x.shape[1] // 2, 2 * x.shape[2], x.shape[3], x.shape[4]), device=x.device, dtype=x.dtype)
    ops.time_unshuffle2_bf16(x, out)
    return out


def add_bias_ncdhw_bf16(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Add channel bias to an NCDHW tensor in place."""

    ops.add_bias_ncdhw_bf16(x, bias)
    return x


def update_cache2_ncdhw_bf16(cur: torch.Tensor, prev: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Update two-frame latent cache from current and previous NCDHW tensors."""

    if out is None:
        out = torch.empty((cur.shape[0], cur.shape[1], 2, cur.shape[3], cur.shape[4]), device=cur.device, dtype=cur.dtype)
    ops.update_cache2_ncdhw_bf16(cur, prev, out)
    return out


__all__ = [
    "ncdhw_to_blc_bf16",
    "patch_im2col_bf16",
    "time_unshuffle2_bf16",
    "add_bias_ncdhw_bf16",
    "update_cache2_ncdhw_bf16",
]
