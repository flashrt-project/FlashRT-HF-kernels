"""FlashRT TurboQuant KV cache kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def _check_packed(k_idx: torch.Tensor, k_qjl: torch.Tensor, v_idx: torch.Tensor) -> int:
    if k_idx.dim() != 2 or k_idx.shape[1] != 128:
        raise RuntimeError("k_idx_packed must have shape (M, 128)")
    m = k_idx.shape[0]
    if k_qjl.shape != (m, 32):
        raise RuntimeError("k_qjl_packed must have shape (M, 32)")
    if v_idx.shape != (m, 128):
        raise RuntimeError("v_idx_packed must have shape (M, 128)")
    return m


@torch.library.register_fake(add_op_namespace_prefix("unpack_packed_bf16"))
def _unpack_packed_bf16_fake(
    k_idx_packed: torch.Tensor,
    k_qjl_packed: torch.Tensor,
    v_idx_packed: torch.Tensor,
    cb_k_mse: torch.Tensor,
    cb_v: torch.Tensor,
    b_k_mse: int,
    b_v: int,
    y_k: torch.Tensor,
    qjl_bf: torch.Tensor,
    y_v: torch.Tensor,
) -> None:
    m = _check_packed(k_idx_packed, k_qjl_packed, v_idx_packed)
    if y_k.shape != (m, 256) or qjl_bf.shape != (m, 256) or y_v.shape != (m, 256):
        raise RuntimeError("outputs must have shape (M, 256)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("unpack_packed_mixed"))
def _unpack_packed_mixed_fake(
    k_idx_packed: torch.Tensor,
    k_qjl_packed: torch.Tensor,
    v_idx_packed: torch.Tensor,
    cb_k_mse: torch.Tensor,
    cb_v: torch.Tensor,
    b_k_mse: int,
    b_v: int,
    y_k: torch.Tensor,
    qjl_f: torch.Tensor,
    y_v: torch.Tensor,
) -> None:
    m = _check_packed(k_idx_packed, k_qjl_packed, v_idx_packed)
    if y_k.shape != (m, 256) or qjl_f.shape != (m, 256) or y_v.shape != (m, 256):
        raise RuntimeError("outputs must have shape (M, 256)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("combine_kv_bf16"))
def _combine_kv_bf16_fake(
    k_mse: torch.Tensor,
    k_qjl: torch.Tensor,
    v_unit: torch.Tensor,
    k_norm: torch.Tensor,
    k_rnorm: torch.Tensor,
    v_norm: torch.Tensor,
    coef: float,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> None:
    if k_mse.dim() != 2 or k_mse.shape[1] != 256:
        raise RuntimeError("k_mse must have shape (M, 256)")
    m = k_mse.shape[0]
    if k_qjl.shape != k_mse.shape or v_unit.shape != k_mse.shape:
        raise RuntimeError("k_qjl and v_unit must have shape (M, 256)")
    if k_norm.shape != (m,) or k_rnorm.shape != (m,) or v_norm.shape != (m,):
        raise RuntimeError("norm tensors must have shape (M,)")
    if k_out.shape != k_mse.shape or v_out.shape != k_mse.shape:
        raise RuntimeError("outputs must have shape (M, 256)")
    return None


def unpack_packed_bf16(
    k_idx_packed: torch.Tensor,
    k_qjl_packed: torch.Tensor,
    v_idx_packed: torch.Tensor,
    cb_k_mse: torch.Tensor,
    cb_v: torch.Tensor,
    b_k_mse: int,
    b_v: int,
    *,
    y_k: Optional[torch.Tensor] = None,
    qjl_bf: Optional[torch.Tensor] = None,
    y_v: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack TurboQuant packed indices to BF16 `(M, 256)` tensors."""

    m = k_idx_packed.shape[0]
    if y_k is None:
        y_k = torch.empty((m, 256), device=k_idx_packed.device, dtype=torch.bfloat16)
    if qjl_bf is None:
        qjl_bf = torch.empty_like(y_k)
    if y_v is None:
        y_v = torch.empty_like(y_k)
    ops.unpack_packed_bf16(
        k_idx_packed, k_qjl_packed, v_idx_packed, cb_k_mse, cb_v,
        int(b_k_mse), int(b_v), y_k, qjl_bf, y_v,
    )
    return y_k, qjl_bf, y_v


def unpack_packed_mixed(
    k_idx_packed: torch.Tensor,
    k_qjl_packed: torch.Tensor,
    v_idx_packed: torch.Tensor,
    cb_k_mse: torch.Tensor,
    cb_v: torch.Tensor,
    b_k_mse: int,
    b_v: int,
    *,
    y_k: Optional[torch.Tensor] = None,
    qjl_f: Optional[torch.Tensor] = None,
    y_v: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack packed K/V to BF16 K/V and FP32 QJL signs."""

    m = k_idx_packed.shape[0]
    if y_k is None:
        y_k = torch.empty((m, 256), device=k_idx_packed.device, dtype=torch.bfloat16)
    if qjl_f is None:
        qjl_f = torch.empty((m, 256), device=k_idx_packed.device, dtype=torch.float32)
    if y_v is None:
        y_v = torch.empty_like(y_k)
    ops.unpack_packed_mixed(
        k_idx_packed, k_qjl_packed, v_idx_packed, cb_k_mse, cb_v,
        int(b_k_mse), int(b_v), y_k, qjl_f, y_v,
    )
    return y_k, qjl_f, y_v


def combine_kv_bf16(
    k_mse: torch.Tensor,
    k_qjl: torch.Tensor,
    v_unit: torch.Tensor,
    k_norm: torch.Tensor,
    k_rnorm: torch.Tensor,
    v_norm: torch.Tensor,
    coef: float,
    *,
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine TurboQuant K/V GEMM outputs into BF16 K/V cache tensors."""

    if k_out is None:
        k_out = torch.empty_like(k_mse)
    if v_out is None:
        v_out = torch.empty_like(v_unit)
    ops.combine_kv_bf16(k_mse, k_qjl, v_unit, k_norm, k_rnorm, v_norm, float(coef), k_out, v_out)
    return k_out, v_out


__all__ = [
    "combine_kv_bf16",
    "unpack_packed_bf16",
    "unpack_packed_mixed",
]
