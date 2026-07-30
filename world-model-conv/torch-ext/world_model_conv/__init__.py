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
    if ci not in (32, 64):
        raise RuntimeError(
            "FP8 Conv3D is accepted only for Ci=32/64; use the "
            "strong-library or NVFP4 path for larger channels"
        )
    if residual.shape != (n, co, t_new, h, w) or out.shape != residual.shape:
        raise RuntimeError("residual/out must be NCDHW")
    if bias.shape != (co,):
        raise RuntimeError("bias must have shape (Co,)")
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("fp8_causal_conv3d_ndhwc_bf16")
)
def _fp8_causal_conv3d_fake(
    cache_x: torch.Tensor,
    new_x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    alpha: float,
    out: torch.Tensor,
) -> None:
    if cache_x.dim() != 5 or new_x.dim() != 5:
        raise RuntimeError("cache_x/new_x must be NDHWC")
    n, t, h, w, ci = new_x.shape
    co = weight.shape[0]
    if cache_x.shape != (n, 2, h, w, ci):
        raise RuntimeError("cache_x must have shape (N,2,H,W,Ci)")
    if weight.shape != (co, 3, 3, 3, ci):
        raise RuntimeError("weight must have shape (Co,3,3,3,Ci)")
    if ci not in (32, 64):
        raise RuntimeError(
            "FP8 Conv3D is accepted only for Ci=32/64; use the "
            "strong-library or NVFP4 path for larger channels"
        )
    if bias.shape != (co,) or out.shape != (n, t, h, w, co):
        raise RuntimeError("bias/out shape mismatch")
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("fp8_conv2d_3x3_nhwc_bf16")
)
def _fp8_conv2d_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    alpha: float,
    out: torch.Tensor,
) -> None:
    if input.dim() != 4:
        raise RuntimeError("input must have shape (N,H,W,Ci)")
    n, h, w, ci = input.shape
    co = weight.shape[0]
    if weight.shape != (co, 3, 3, ci):
        raise RuntimeError("weight must have shape (Co,3,3,Ci)")
    if bias.shape != (co,) or out.shape != (n, h, w, co):
        raise RuntimeError("bias/out shape mismatch")
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("fp8_conv2d_3x3_ncdhw_bf16")
)
def _fp8_conv2d_ncdhw_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    alpha: float,
    out: torch.Tensor,
) -> None:
    if input.dim() != 5:
        raise RuntimeError("input must have shape (B,T,H,W,Ci)")
    b, t, h, w, ci = input.shape
    co = weight.shape[0]
    if weight.shape != (co, 3, 3, ci):
        raise RuntimeError("weight must have shape (Co,3,3,Ci)")
    if bias.shape != (co,) or out.shape != (b, co, t, h, w):
        raise RuntimeError("bias/out shape mismatch")
    return None


def _check_nvfp4_conv_shapes(
    cache_packed: torch.Tensor,
    new_packed: torch.Tensor,
    weight_packed: torch.Tensor,
    cache_sf: torch.Tensor,
    new_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    bias: torch.Tensor,
    outer_weight: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if (
        cache_packed.dim() != 5
        or new_packed.dim() != 5
        or weight_packed.dim() != 5
    ):
        raise RuntimeError("packed inputs must be five-dimensional")
    n, t, h, w, ci_half = new_packed.shape
    ci = ci_half * 2
    co = weight_packed.shape[0]
    if (
        (ci != 64 and ci % 128 != 0)
        or co % 8 != 0
        or cache_packed.shape != (n, 2, h, w, ci // 2)
        or weight_packed.shape != (co, 3, 3, 3, ci // 2)
        or cache_sf.shape != (n, 2, h, w, ci // 16)
        or new_sf.shape != (n, t, h, w, ci // 16)
        or weight_sf.shape != (co, 3, 3, 3, ci // 16)
        or bias.shape != (co,)
        or (outer_weight is not None and outer_weight.shape != (co,))
    ):
        raise RuntimeError(
            "NVFP4 Conv3D accepts Ci=64 or multiples of 128; "
            "other shapes must use the strong-library path"
        )
    return n, t, h, w, co


@torch.library.register_fake(
    add_op_namespace_prefix("nvfp4_causal_conv3d_ndhwc_bf16")
)
def _nvfp4_causal_conv3d_fake(
    cache_packed: torch.Tensor,
    new_packed: torch.Tensor,
    weight_packed: torch.Tensor,
    cache_sf: torch.Tensor,
    new_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    bias: torch.Tensor,
    outer_weight: torch.Tensor | None,
    alpha: float,
    out: torch.Tensor,
) -> None:
    del alpha
    n, t, h, w, co = _check_nvfp4_conv_shapes(
        cache_packed,
        new_packed,
        weight_packed,
        cache_sf,
        new_sf,
        weight_sf,
        bias,
        outer_weight,
    )
    if out.shape != (n, t, h, w, co):
        raise RuntimeError("out must have shape (N,T,H,W,Co)")
    return None


@torch.library.register_fake(
    add_op_namespace_prefix(
        "nvfp4_causal_conv3d_residual_ncdhw_bf16"
    )
)
def _nvfp4_causal_conv3d_residual_fake(
    cache_packed: torch.Tensor,
    new_packed: torch.Tensor,
    weight_packed: torch.Tensor,
    cache_sf: torch.Tensor,
    new_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    outer_weight: torch.Tensor | None,
    alpha: float,
    out: torch.Tensor,
) -> None:
    del alpha
    n, t, h, w, co = _check_nvfp4_conv_shapes(
        cache_packed,
        new_packed,
        weight_packed,
        cache_sf,
        new_sf,
        weight_sf,
        bias,
        outer_weight,
    )
    if residual.shape != (n, co, t, h, w) or out.shape != residual.shape:
        raise RuntimeError("residual/out must have shape (N,Co,T,H,W)")
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


def fp8_causal_conv3d_ndhwc_bf16(
    cache_x: torch.Tensor,
    new_x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    alpha: float = 1.0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 causal 3D convolution with virtual two-frame cache concat."""

    n, t, h, w, _ = new_x.shape
    if out is None:
        out = torch.empty(
            (n, t, h, w, weight.shape[0]),
            device=new_x.device,
            dtype=torch.bfloat16,
        )
    ops.fp8_causal_conv3d_ndhwc_bf16(
        cache_x, new_x, weight, bias, float(alpha), out
    )
    return out


def fp8_conv2d_3x3_nhwc_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    alpha: float = 1.0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 3x3 Conv2D with NHWC input/output and BF16 epilogue."""

    if out is None:
        out = torch.empty(
            (*input.shape[:3], weight.shape[0]),
            device=input.device,
            dtype=torch.bfloat16,
        )
    ops.fp8_conv2d_3x3_nhwc_bf16(
        input, weight, bias, float(alpha), out
    )
    return out


def fp8_conv2d_3x3_ncdhw_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    alpha: float = 1.0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 3x3 Conv2D over B*T frames with direct BF16 NCDHW output."""

    if out is None:
        out = torch.empty(
            (
                input.shape[0],
                weight.shape[0],
                input.shape[1],
                input.shape[2],
                input.shape[3],
            ),
            device=input.device,
            dtype=torch.bfloat16,
        )
    ops.fp8_conv2d_3x3_ncdhw_bf16(
        input, weight, bias, float(alpha), out
    )
    return out


def nvfp4_causal_conv3d_ndhwc_bf16(
    cache_packed: torch.Tensor,
    new_packed: torch.Tensor,
    weight_packed: torch.Tensor,
    cache_sf: torch.Tensor,
    new_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    bias: torch.Tensor,
    outer_weight: torch.Tensor | None = None,
    alpha: float = 1.0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NVFP4 causal Conv3D with linear UE4M3 scale factors and BF16 NDHWC output."""

    n, t, h, w, _ = new_packed.shape
    if out is None:
        out = torch.empty(
            (n, t, h, w, weight_packed.shape[0]),
            device=new_packed.device,
            dtype=torch.bfloat16,
        )
    ops.nvfp4_causal_conv3d_ndhwc_bf16(
        cache_packed,
        new_packed,
        weight_packed,
        cache_sf,
        new_sf,
        weight_sf,
        bias,
        outer_weight,
        float(alpha),
        out,
    )
    return out


def nvfp4_causal_conv3d_residual_ncdhw_bf16(
    cache_packed: torch.Tensor,
    new_packed: torch.Tensor,
    weight_packed: torch.Tensor,
    cache_sf: torch.Tensor,
    new_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    outer_weight: torch.Tensor | None = None,
    alpha: float = 1.0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NVFP4 causal Conv3D with fused bias, residual, and BF16 NCDHW output."""

    if out is None:
        out = torch.empty_like(residual)
    ops.nvfp4_causal_conv3d_residual_ncdhw_bf16(
        cache_packed,
        new_packed,
        weight_packed,
        cache_sf,
        new_sf,
        weight_sf,
        bias,
        residual,
        outer_weight,
        float(alpha),
        out,
    )
    return out


__all__ = [
    "fp8_conv3d_v18_ncdhw_res_bf16out",
    "fp8_causal_conv3d_ndhwc_bf16",
    "fp8_conv2d_3x3_nhwc_bf16",
    "fp8_conv2d_3x3_ncdhw_bf16",
    "nvfp4_causal_conv3d_ndhwc_bf16",
    "nvfp4_causal_conv3d_residual_ncdhw_bf16",
]
