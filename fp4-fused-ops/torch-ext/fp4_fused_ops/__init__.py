"""FlashRT FP4 fused producer kernels."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


def sfa_size_bytes(rows: int, dim: int, is_sfb: bool = False, device: torch.device | str | None = None) -> int:
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    anchor = torch.empty((1,), device=device, dtype=torch.uint8)
    return int(ops.sfa_size_bytes_for(anchor, int(rows), int(dim), bool(is_sfb)))


def _alloc_fp4(rows: int, dim: int, device: torch.device | str):
    packed = torch.empty((rows, dim // 2), device=device, dtype=torch.uint8)
    sfa = torch.empty((sfa_size_bytes(rows, dim, False, device=device),), device=device, dtype=torch.uint8)
    return packed, sfa


@torch.library.register_fake(add_op_namespace_prefix("rms_norm_fp4_sfa_fp16"))
def _rms_norm_fake(x: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("residual_add_rms_norm_fp4_sfa_fp16"))
def _residual_fake(residual: torch.Tensor, x: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("residual_add_rms_norm_fp4_sfa_v2_fp16"))
def _residual_v2_fake(residual: torch.Tensor, x: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("residual_add_rms_norm_mul_fp4_sfa_fp16"))
def _residual_mul_fake(
    residual: torch.Tensor,
    x: torch.Tensor,
    inv_s: torch.Tensor,
    packed: torch.Tensor,
    sfa: torch.Tensor,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("silu_mul_fp4_sfa_fp16"))
def _silu_fake(merged: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("silu_mul_fp4_sfa_v2_fp16"))
def _silu_v2_fake(merged: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("silu_mul_mul_fp4_sfa_v2_fp16"))
def _silu_mul_v2_fake(merged: torch.Tensor, inv_s: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("silu_mul_two_fp4_to_fp4"))
def _silu_two_fake(
    gate_packed: torch.Tensor,
    gate_sfa: torch.Tensor,
    up_packed: torch.Tensor,
    up_sfa: torch.Tensor,
    out_packed: torch.Tensor,
    out_sfa: torch.Tensor,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("silu_mul_two_mul_fp4_to_fp4"))
def _silu_two_mul_fake(
    gate_packed: torch.Tensor,
    gate_sfa: torch.Tensor,
    up_packed: torch.Tensor,
    up_sfa: torch.Tensor,
    inv_s: torch.Tensor,
    out_packed: torch.Tensor,
    out_sfa: torch.Tensor,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("dequantize_fp4_sfa_fp16"))
def _dequant_fake(packed: torch.Tensor, sfa: torch.Tensor, out: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("rms_silu_nvfp4_ndhwc_bf16"))
def _rms_silu_nvfp4_ndhwc_bf16_fake(
    x: torch.Tensor,
    gamma: torch.Tensor,
    awq_inv_scale: torch.Tensor | None,
    eps: float,
    packed: torch.Tensor,
    scale_factors: torch.Tensor,
) -> None:
    del eps
    if x.dim() != 5:
        raise RuntimeError("x must have shape (B,C,T,H,W)")
    b, c, t, h, w = x.shape
    if (
        c % 128 != 0
        or c > 1024
        or gamma.shape != (c,)
        or packed.shape != (b, t, h, w, c // 2)
        or scale_factors.shape != (b, t, h, w, c // 16)
        or (awq_inv_scale is not None and awq_inv_scale.shape != (c,))
    ):
        raise RuntimeError(
            "rms_silu_nvfp4_ndhwc_bf16 shape contract failed"
        )
    return None


@torch.library.register_fake(
    add_op_namespace_prefix("quantize_bf16_to_nvfp4_linear")
)
def _quantize_bf16_to_nvfp4_linear_fake(
    input: torch.Tensor,
    packed: torch.Tensor,
    scale_factors: torch.Tensor,
) -> None:
    if input.dim() != 2:
        raise RuntimeError("input must have shape (rows, cols)")
    rows, cols = input.shape
    if (
        rows <= 0
        or cols <= 0
        or cols % 16 != 0
        or packed.shape != (rows, cols // 2)
        or scale_factors.shape != (rows, cols // 16)
    ):
        raise RuntimeError(
            "quantize_bf16_to_nvfp4_linear shape contract failed"
        )
    return None


def rms_norm_fp4_sfa_fp16(x: torch.Tensor, packed: torch.Tensor | None = None, sfa: torch.Tensor | None = None):
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(x.shape[0], x.shape[1], x.device)
    ops.rms_norm_fp4_sfa_fp16(x, packed, sfa)
    return packed, sfa


def residual_add_rms_norm_fp4_sfa_fp16(
    residual: torch.Tensor,
    x: torch.Tensor,
    packed: torch.Tensor | None = None,
    sfa: torch.Tensor | None = None,
):
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(x.shape[0], x.shape[1], x.device)
    ops.residual_add_rms_norm_fp4_sfa_fp16(residual, x, packed, sfa)
    return packed, sfa


def residual_add_rms_norm_fp4_sfa_v2_fp16(
    residual: torch.Tensor,
    x: torch.Tensor,
    packed: torch.Tensor | None = None,
    sfa: torch.Tensor | None = None,
):
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(x.shape[0], x.shape[1], x.device)
    ops.residual_add_rms_norm_fp4_sfa_v2_fp16(residual, x, packed, sfa)
    return packed, sfa


def residual_add_rms_norm_mul_fp4_sfa_fp16(
    residual: torch.Tensor,
    x: torch.Tensor,
    inv_s: torch.Tensor,
    packed: torch.Tensor | None = None,
    sfa: torch.Tensor | None = None,
):
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(x.shape[0], x.shape[1], x.device)
    ops.residual_add_rms_norm_mul_fp4_sfa_fp16(residual, x, inv_s, packed, sfa)
    return packed, sfa


def silu_mul_fp4_sfa_fp16(merged: torch.Tensor, packed: torch.Tensor | None = None, sfa: torch.Tensor | None = None):
    hidden = merged.shape[1] // 2
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(merged.shape[0], hidden, merged.device)
    ops.silu_mul_fp4_sfa_fp16(merged, packed, sfa)
    return packed, sfa


def silu_mul_fp4_sfa_v2_fp16(merged: torch.Tensor, packed: torch.Tensor | None = None, sfa: torch.Tensor | None = None):
    hidden = merged.shape[1] // 2
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(merged.shape[0], hidden, merged.device)
    ops.silu_mul_fp4_sfa_v2_fp16(merged, packed, sfa)
    return packed, sfa


def silu_mul_mul_fp4_sfa_v2_fp16(
    merged: torch.Tensor,
    inv_s: torch.Tensor,
    packed: torch.Tensor | None = None,
    sfa: torch.Tensor | None = None,
):
    hidden = merged.shape[1] // 2
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(merged.shape[0], hidden, merged.device)
    ops.silu_mul_mul_fp4_sfa_v2_fp16(merged, inv_s, packed, sfa)
    return packed, sfa


def silu_mul_two_fp4_to_fp4(
    gate_packed: torch.Tensor,
    gate_sfa: torch.Tensor,
    up_packed: torch.Tensor,
    up_sfa: torch.Tensor,
    out_packed: torch.Tensor | None = None,
    out_sfa: torch.Tensor | None = None,
):
    hidden = gate_packed.shape[1] * 2
    if out_packed is None or out_sfa is None:
        out_packed, out_sfa = _alloc_fp4(gate_packed.shape[0], hidden, gate_packed.device)
    ops.silu_mul_two_fp4_to_fp4(gate_packed, gate_sfa, up_packed, up_sfa, out_packed, out_sfa)
    return out_packed, out_sfa


def silu_mul_two_mul_fp4_to_fp4(
    gate_packed: torch.Tensor,
    gate_sfa: torch.Tensor,
    up_packed: torch.Tensor,
    up_sfa: torch.Tensor,
    inv_s: torch.Tensor,
    out_packed: torch.Tensor | None = None,
    out_sfa: torch.Tensor | None = None,
):
    hidden = gate_packed.shape[1] * 2
    if out_packed is None or out_sfa is None:
        out_packed, out_sfa = _alloc_fp4(gate_packed.shape[0], hidden, gate_packed.device)
    ops.silu_mul_two_mul_fp4_to_fp4(gate_packed, gate_sfa, up_packed, up_sfa, inv_s, out_packed, out_sfa)
    return out_packed, out_sfa


def dequantize_fp4_sfa_fp16(
    packed: torch.Tensor,
    sfa: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((packed.shape[0], packed.shape[1] * 2), device=packed.device, dtype=torch.float16)
    ops.dequantize_fp4_sfa_fp16(packed, sfa, out)
    return out


def rms_silu_nvfp4_ndhwc_bf16(
    x: torch.Tensor,
    gamma: torch.Tensor,
    awq_inv_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
    *,
    packed: torch.Tensor | None = None,
    scale_factors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse BF16 NCDHW RMSNorm, SiLU and linear-SF NVFP4 quantization."""

    b, c, t, h, w = x.shape
    if packed is None:
        packed = torch.empty(
            (b, t, h, w, c // 2), device=x.device, dtype=torch.uint8
        )
    if scale_factors is None:
        scale_factors = torch.empty(
            (b, t, h, w, c // 16), device=x.device, dtype=torch.uint8
        )
    ops.rms_silu_nvfp4_ndhwc_bf16(
        x,
        gamma,
        awq_inv_scale,
        float(eps),
        packed,
        scale_factors,
    )
    return packed, scale_factors


def quantize_bf16_to_nvfp4_linear(
    input: torch.Tensor,
    *,
    packed: torch.Tensor | None = None,
    scale_factors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 rows to packed E2M1 with linear UE4M3 block-16 scales."""

    rows, cols = input.shape
    if packed is None:
        packed = torch.empty(
            (rows, cols // 2), device=input.device, dtype=torch.uint8
        )
    if scale_factors is None:
        scale_factors = torch.empty(
            (rows, cols // 16), device=input.device, dtype=torch.uint8
        )
    ops.quantize_bf16_to_nvfp4_linear(input, packed, scale_factors)
    return packed, scale_factors


def _check_ncdhw(x: torch.Tensor, gamma: torch.Tensor) -> None:
    if x.dim() != 5:
        raise RuntimeError("x must have shape (B,C,T,H,W)")
    if x.shape[1] <= 0 or x.shape[1] % 2 != 0 or x.shape[1] > 1024:
        raise RuntimeError("C must be even and at most 1024")
    if gamma.shape != (x.shape[1],):
        raise RuntimeError("gamma must have shape (C,)")


@torch.library.register_fake(add_op_namespace_prefix("bf16_rms_silu_ncdhw"))
def _bf16_rms_silu_ncdhw_fake(
    x: torch.Tensor,
    gamma: torch.Tensor,
    prev_cache: torch.Tensor | None,
    eps: float,
    out: torch.Tensor,
    next_cache: torch.Tensor | None,
) -> None:
    del eps
    _check_ncdhw(x, gamma)
    if out.shape != x.shape:
        raise RuntimeError("out must have the same shape as x")
    expected_cache = (x.shape[0], x.shape[1], 2, x.shape[3], x.shape[4])
    if prev_cache is not None and prev_cache.shape != expected_cache:
        raise RuntimeError("prev_cache must have shape (B,C,2,H,W)")
    if next_cache is not None and next_cache.shape != expected_cache:
        raise RuntimeError("next_cache must have shape (B,C,2,H,W)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("bf16_rms_norm_ncdhw"))
def _bf16_rms_norm_ncdhw_fake(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    out: torch.Tensor,
) -> None:
    del eps
    _check_ncdhw(x, gamma)
    if bias is not None and bias.shape != (x.shape[1],):
        raise RuntimeError("bias must have shape (C,)")
    if out.shape != x.shape:
        raise RuntimeError("out must have the same shape as x")
    return None


def bf16_rms_silu_ncdhw(
    x: torch.Tensor,
    gamma: torch.Tensor,
    prev_cache: torch.Tensor | None = None,
    eps: float = 1e-6,
    *,
    out: torch.Tensor | None = None,
    next_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply channel RMSNorm and SiLU to BF16 NCDHW with optional cache update."""

    if out is None:
        out = torch.empty_like(x)
    ops.bf16_rms_silu_ncdhw(
        x, gamma, prev_cache, float(eps), out, next_cache
    )
    return out, next_cache


def bf16_rms_norm_ncdhw(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply channel RMSNorm and optional bias to BF16 NCDHW."""

    if out is None:
        out = torch.empty_like(x)
    ops.bf16_rms_norm_ncdhw(x, gamma, bias, float(eps), out)
    return out


__all__ = [
    "bf16_rms_norm_ncdhw",
    "bf16_rms_silu_ncdhw",
    "dequantize_fp4_sfa_fp16",
    "quantize_bf16_to_nvfp4_linear",
    "residual_add_rms_norm_fp4_sfa_fp16",
    "residual_add_rms_norm_fp4_sfa_v2_fp16",
    "residual_add_rms_norm_mul_fp4_sfa_fp16",
    "rms_norm_fp4_sfa_fp16",
    "rms_silu_nvfp4_ndhwc_bf16",
    "sfa_size_bytes",
    "silu_mul_fp4_sfa_fp16",
    "silu_mul_fp4_sfa_v2_fp16",
    "silu_mul_mul_fp4_sfa_v2_fp16",
    "silu_mul_two_fp4_to_fp4",
    "silu_mul_two_mul_fp4_to_fp4",
]
