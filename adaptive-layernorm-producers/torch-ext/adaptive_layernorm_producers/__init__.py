"""FlashRT adaptive LayerNorm producer kernels for DiT/Wan-style blocks."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_x(x: torch.Tensor) -> None:
    if x.dim() != 2:
        raise RuntimeError("x must have shape (rows, dim)")
    if x.shape[1] % 2 != 0:
        raise RuntimeError("x.shape[1] must be even")


def _check_mod(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> None:
    _check_x(x)
    dim = x.shape[1]
    if scale.shape != (dim,) or shift.shape != (dim,):
        raise RuntimeError("scale and shift must have shape (dim,)")


def _check_scalar(scale: torch.Tensor, name: str) -> None:
    if scale.numel() != 1:
        raise RuntimeError(f"{name} must be a scalar tensor")


def swizzled_sf_size(rows: int, dim: int) -> int:
    """Return bytes required by the FlashRT/CUTLASS NVFP4 swizzled scale layout."""

    if dim % 16 != 0:
        raise RuntimeError("dim must be divisible by 16 for NVFP4 swizzled output")
    n_blocks = dim // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 128 * 64


def _alloc_fp8(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.float8_e4m3fn)


def _alloc_nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, dim = x.shape
    packed = torch.empty((rows, dim // 2), device=x.device, dtype=torch.uint8)
    sf = torch.zeros((swizzled_sf_size(rows, dim),), device=x.device, dtype=torch.uint8)
    return packed, sf


@torch.library.register_fake(add_op_namespace_prefix("ada_layer_norm_quant_fp8_bf16"))
def _ada_layer_norm_quant_fp8_bf16_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    act_scale: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    _check_mod(x, scale, shift)
    _check_scalar(act_scale, "act_scale")
    if out.shape != x.shape:
        raise RuntimeError("out must have the same shape as x")
    return None


@torch.library.register_fake(add_op_namespace_prefix("ada_layer_norm_quant_fp8_modfp8_bf16"))
def _ada_layer_norm_quant_fp8_modfp8_bf16_fake(
    x: torch.Tensor,
    scale_fp8: torch.Tensor,
    shift_fp8: torch.Tensor,
    scale_deq: torch.Tensor,
    shift_deq: torch.Tensor,
    act_scale: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    _check_mod(x, scale_fp8, shift_fp8)
    _check_scalar(scale_deq, "scale_deq")
    _check_scalar(shift_deq, "shift_deq")
    _check_scalar(act_scale, "act_scale")
    if out.shape != x.shape:
        raise RuntimeError("out must have the same shape as x")
    return None


@torch.library.register_fake(add_op_namespace_prefix("awq_ada_layer_norm_quant_fp8_bf16"))
def _awq_ada_layer_norm_quant_fp8_bf16_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    inv_s: torch.Tensor,
    act_scale: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    _check_mod(x, scale, shift)
    if inv_s.shape != (x.shape[1],):
        raise RuntimeError("inv_s must have shape (dim,)")
    _check_scalar(act_scale, "act_scale")
    if out.shape != x.shape:
        raise RuntimeError("out must have the same shape as x")
    return None


@torch.library.register_fake(add_op_namespace_prefix("ada_layer_norm_quant_nvfp4_swizzled_bf16"))
def _ada_layer_norm_quant_nvfp4_swizzled_bf16_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    packed: torch.Tensor,
    sf_swizzled: torch.Tensor,
) -> None:
    _check_mod(x, scale, shift)
    rows, dim = x.shape
    if packed.shape != (rows, dim // 2):
        raise RuntimeError("packed must have shape (rows, dim // 2)")
    if sf_swizzled.numel() < swizzled_sf_size(rows, dim):
        raise RuntimeError("sf_swizzled is too small")
    return None


@torch.library.register_fake(add_op_namespace_prefix("ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16"))
def _ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16_fake(
    x: torch.Tensor,
    scale_fp8: torch.Tensor,
    shift_fp8: torch.Tensor,
    scale_deq: torch.Tensor,
    shift_deq: torch.Tensor,
    eps: float,
    packed: torch.Tensor,
    sf_swizzled: torch.Tensor,
) -> None:
    _check_mod(x, scale_fp8, shift_fp8)
    _check_scalar(scale_deq, "scale_deq")
    _check_scalar(shift_deq, "shift_deq")
    rows, dim = x.shape
    if packed.shape != (rows, dim // 2):
        raise RuntimeError("packed must have shape (rows, dim // 2)")
    if sf_swizzled.numel() < swizzled_sf_size(rows, dim):
        raise RuntimeError("sf_swizzled is too small")
    return None


@torch.library.register_fake(add_op_namespace_prefix("layer_norm_no_affine_quant_fp8_static_bf16"))
def _layer_norm_no_affine_quant_fp8_static_bf16_fake(
    x: torch.Tensor,
    act_scale: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    _check_x(x)
    _check_scalar(act_scale, "act_scale")
    if out.shape != x.shape:
        raise RuntimeError("out must have the same shape as x")
    return None


def ada_layer_norm_quant_fp8_bf16(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    act_scale: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused no-affine LayerNorm + AdaLN scale/shift + static FP8 quantize."""

    if out is None:
        out = _alloc_fp8(x)
    ops.ada_layer_norm_quant_fp8_bf16(x, scale, shift, act_scale, float(eps), out)
    return out


def ada_layer_norm_quant_fp8_modfp8_bf16(
    x: torch.Tensor,
    scale_fp8: torch.Tensor,
    shift_fp8: torch.Tensor,
    scale_deq: torch.Tensor,
    shift_deq: torch.Tensor,
    act_scale: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused AdaLN producer when modulation vectors are stored in FP8."""

    if out is None:
        out = _alloc_fp8(x)
    ops.ada_layer_norm_quant_fp8_modfp8_bf16(
        x, scale_fp8, shift_fp8, scale_deq, shift_deq, act_scale, float(eps), out
    )
    return out


def awq_ada_layer_norm_quant_fp8_bf16(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    inv_s: torch.Tensor,
    act_scale: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused AdaLN producer with AWQ/SmoothQuant per-channel activation scale."""

    if out is None:
        out = _alloc_fp8(x)
    ops.awq_ada_layer_norm_quant_fp8_bf16(x, scale, shift, inv_s, act_scale, float(eps), out)
    return out


def ada_layer_norm_quant_nvfp4_swizzled_bf16(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
    packed: torch.Tensor | None = None,
    sf_swizzled: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused AdaLN producer to NVFP4 packed activations and swizzled scale factors."""

    if packed is None or sf_swizzled is None:
        packed, sf_swizzled = _alloc_nvfp4(x)
    ops.ada_layer_norm_quant_nvfp4_swizzled_bf16(x, scale, shift, float(eps), packed, sf_swizzled)
    return packed, sf_swizzled


def ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16(
    x: torch.Tensor,
    scale_fp8: torch.Tensor,
    shift_fp8: torch.Tensor,
    scale_deq: torch.Tensor,
    shift_deq: torch.Tensor,
    eps: float = 1e-5,
    packed: torch.Tensor | None = None,
    sf_swizzled: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused FP8-modulated AdaLN producer to NVFP4 packed activations."""

    if packed is None or sf_swizzled is None:
        packed, sf_swizzled = _alloc_nvfp4(x)
    ops.ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16(
        x, scale_fp8, shift_fp8, scale_deq, shift_deq, float(eps), packed, sf_swizzled
    )
    return packed, sf_swizzled


def layer_norm_no_affine_quant_fp8_static_bf16(
    x: torch.Tensor,
    act_scale: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused no-affine LayerNorm + static FP8 quantize."""

    if out is None:
        out = _alloc_fp8(x)
    ops.layer_norm_no_affine_quant_fp8_static_bf16(x, act_scale, float(eps), out)
    return out


__all__ = [
    "ada_layer_norm_quant_fp8_bf16",
    "ada_layer_norm_quant_fp8_modfp8_bf16",
    "awq_ada_layer_norm_quant_fp8_bf16",
    "ada_layer_norm_quant_nvfp4_swizzled_bf16",
    "ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16",
    "layer_norm_no_affine_quant_fp8_static_bf16",
    "swizzled_sf_size",
]
