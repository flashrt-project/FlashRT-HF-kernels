"""FlashRT world-model convolution kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("fp8_conv3d_v18_ncdhw_res_bf16out"))
def _fp8_conv3d_fake(
    cache_x: torch.Tensor,
    new_x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    alpha: float,
    out: torch.Tensor,
) -> None:
    if cache_x.dim() != 5 or new_x.dim() != 5:
        raise RuntimeError("cache_x/new_x must be NDHWC")
    n, t_new, h, w, ci = new_x.shape
    co = weight.shape[0]
    if cache_x.shape != (n, 2, h, w, ci):
        raise RuntimeError("cache_x must have shape (N,2,H,W,Ci)")
    if weight.shape != (co, 3, 3, 3, ci):
        raise RuntimeError("weight must have shape (Co,3,3,3,Ci)")
    if residual.shape != (n, co, t_new, h, w) or out.shape != residual.shape:
        raise RuntimeError("residual/out must be NCDHW")
    if bias.shape != (co,):
        raise RuntimeError("bias must have shape (Co,)")
    return None


def fp8_conv3d_v18_ncdhw_res_bf16out(
    cache_x: torch.Tensor,
    new_x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    alpha: float = 1.0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 3D causal conv with virtual cache concat, bias, residual, BF16 NCDHW output."""

    n, t_new, h, w, _ = new_x.shape
    co = weight.shape[0]
    if out is None:
        out = torch.empty((n, co, t_new, h, w), device=new_x.device, dtype=torch.bfloat16)
    ops.fp8_conv3d_v18_ncdhw_res_bf16out(cache_x, new_x, weight, bias, residual, float(alpha), out)
    return out


__all__ = ["fp8_conv3d_v18_ncdhw_res_bf16out"]
