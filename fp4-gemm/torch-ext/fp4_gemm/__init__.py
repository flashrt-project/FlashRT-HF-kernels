"""FlashRT FP4 GEMM kernels."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


def sfa_size_bytes(rows: int, dim: int) -> int:
    if rows <= 0 or dim <= 0 or dim % 16 != 0:
        raise ValueError("rows must be positive and dim must be positive/divisible by 16")
    n_blocks = dim // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 512


def _alloc_fp4(rows: int, dim: int, device: torch.device | str):
    return (
        torch.empty((rows, dim // 2), device=device, dtype=torch.uint8),
        torch.empty((sfa_size_bytes(rows, dim),), device=device, dtype=torch.uint8),
    )


@torch.library.register_fake(add_op_namespace_prefix("fp4_w4a16_linear_bf16"))
def _linear_fake(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    out: torch.Tensor,
    alpha: float = 1.0,
    variant: int = 0,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_fp4_sfa_fp16"))
def _quant_fake(x: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor, is_sfb: bool = False) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("dequantize_fp4_sfa_fp16"))
def _dequant_fake(packed: torch.Tensor, sfa: torch.Tensor, out: torch.Tensor, is_sfb: bool = False) -> None:
    return None


def quantize_fp4_sfa_fp16(
    x: torch.Tensor,
    packed: torch.Tensor | None = None,
    sfa: torch.Tensor | None = None,
    is_sfb: bool = False,
):
    if packed is None or sfa is None:
        packed, sfa = _alloc_fp4(x.shape[0], x.shape[1], x.device)
    ops.quantize_fp4_sfa_fp16(x, packed, sfa, bool(is_sfb))
    return packed, sfa


def dequantize_fp4_sfa_fp16(
    packed: torch.Tensor,
    sfa: torch.Tensor,
    out: torch.Tensor | None = None,
    is_sfb: bool = False,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((packed.shape[0], packed.shape[1] * 2), device=packed.device, dtype=torch.float16)
    ops.dequantize_fp4_sfa_fp16(packed, sfa, out, bool(is_sfb))
    return out


def fp4_w4a16_linear_bf16(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: float = 1.0,
    out: torch.Tensor | None = None,
    variant: int = 0,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((a_packed.shape[0], b_packed.shape[0]), device=a_packed.device, dtype=torch.bfloat16)
    ops.fp4_w4a16_linear_bf16(a_packed, b_packed, sfa, sfb, out, float(alpha), int(variant))
    return out

