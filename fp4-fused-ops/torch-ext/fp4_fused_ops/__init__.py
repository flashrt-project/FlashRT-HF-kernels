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
