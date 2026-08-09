"""FlashRT grouped MoE GEMV kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(add_op_namespace_prefix("w4a16_decode_gemv_bf16"))
def _w4a16_decode_gemv_fake(
    x_bf16: torch.Tensor,
    weight_packed: torch.Tensor,
    sfb: torch.Tensor,
    alpha: float,
    out: torch.Tensor,
) -> None:
    k = x_bf16.shape[0] if x_bf16.dim() == 1 else x_bf16.shape[1]
    if weight_packed.dim() != 2 or weight_packed.shape[1] != k // 2 or out.shape != (weight_packed.shape[0],):
        raise RuntimeError("expected x (K,) or (1,K), weight_packed (N,K/2), out (N,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("grouped_w4a16_gemv_bf16"))
def _grouped_w4a16_gemv_fake(
    activations: torch.Tensor,
    weight_stack: torch.Tensor,
    sfb_stack: torch.Tensor,
    alpha_stack: torch.Tensor,
    expert_idx: torch.Tensor,
    w_stride: int,
    sfb_stride: int,
    out: torch.Tensor,
) -> None:
    if activations.dim() != 2 or out.dim() != 2 or out.shape[0] != activations.shape[0]:
        raise RuntimeError("expected activations (slots,K), out (slots,N)")
    if expert_idx.shape != (activations.shape[0],):
        raise RuntimeError("expert_idx must have shape (slots,)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_activations_nvfp4_bf16"))
def _quantize_activations_nvfp4_fake(
    activations: torch.Tensor,
    packed: torch.Tensor,
    sfa: torch.Tensor,
) -> None:
    if activations.dim() != 2:
        raise RuntimeError("activations must have shape (M,K)")
    if packed.shape != (activations.shape[0], activations.shape[1] // 2):
        raise RuntimeError("packed must have shape (M,K/2)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_weights_nvfp4_bf16"))
def _quantize_weights_nvfp4_fake(
    weights: torch.Tensor,
    packed: torch.Tensor,
    sfb: torch.Tensor,
) -> None:
    if weights.dim() != 2 or packed.shape != (weights.shape[0], weights.shape[1] // 2):
        raise RuntimeError("expected weights (N,K), packed (N,K/2)")
    return None


@torch.library.register_fake(add_op_namespace_prefix("grouped_w4a4_gemv_bf16"))
def _grouped_w4a4_gemv_fake(
    activations_packed: torch.Tensor,
    weight_stack: torch.Tensor,
    sfa: torch.Tensor,
    sfb_stack: torch.Tensor,
    alpha_stack: torch.Tensor,
    expert_idx: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if activations_packed.dim() != 2 or weight_stack.dim() != 3:
        raise RuntimeError("expected activations_packed (M,K/2), weight_stack (E,N,K/2)")
    if expert_idx.dim() != 2:
        raise RuntimeError("expert_idx must have shape (M,top_k)")
    expected = (activations_packed.shape[0], expert_idx.shape[1], weight_stack.shape[1])
    if out.shape != expected:
        raise RuntimeError(f"out must have shape {expected}")
    return None


def _swizzled_sf_bytes(rows: int, dim: int) -> int:
    return ((int(rows) + 127) // 128) * (((int(dim) // 16) + 3) // 4) * 512


def w4a16_decode_gemv_bf16(
    x_bf16: torch.Tensor,
    weight_packed: torch.Tensor,
    sfb: torch.Tensor,
    *,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((weight_packed.shape[0],), device=x_bf16.device, dtype=torch.bfloat16)
    ops.w4a16_decode_gemv_bf16(x_bf16, weight_packed, sfb, float(alpha), out)
    return out


def grouped_w4a16_gemv_bf16(
    activations: torch.Tensor,
    weight_stack: torch.Tensor,
    sfb_stack: torch.Tensor,
    alpha_stack: torch.Tensor,
    expert_idx: torch.Tensor,
    *,
    n: int,
    w_stride: Optional[int] = None,
    sfb_stride: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one W4A16 GEMV per routed slot.

    `weight_stack` is a flat expert stack. `w_stride` and `sfb_stride` are byte
    strides between experts; by default `w_stride = n * K / 2`.
    """

    k = activations.shape[1]
    if out is None:
        out = torch.empty((activations.shape[0], int(n)), device=activations.device, dtype=torch.bfloat16)
    if w_stride is None:
        w_stride = int(n) * k // 2
    if sfb_stride is None:
        raise RuntimeError("sfb_stride must be provided because swizzled SF size is layout-dependent")
    ops.grouped_w4a16_gemv_bf16(
        activations,
        weight_stack,
        sfb_stack,
        alpha_stack,
        expert_idx,
        int(w_stride),
        int(sfb_stride),
        out,
    )
    return out


def quantize_activations_nvfp4_bf16(
    activations: torch.Tensor,
    *,
    packed: Optional[torch.Tensor] = None,
    sfa: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 ``[M,K]`` activation once for routed W4A4 GEMV.

    Pass preallocated ``packed`` and ``sfa`` buffers on CUDA Graph hot paths.
    """

    m, k = activations.shape
    if packed is None:
        packed = torch.empty((m, k // 2), device=activations.device, dtype=torch.uint8)
    if sfa is None:
        sfa = torch.empty((_swizzled_sf_bytes(m, k),), device=activations.device, dtype=torch.uint8)
    ops.quantize_activations_nvfp4_bf16(activations, packed, sfa)
    return packed, sfa


def quantize_weights_nvfp4_bf16(
    weights: torch.Tensor,
    *,
    packed: Optional[torch.Tensor] = None,
    sfb: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline/helper quantization for one expert's BF16 ``[N,K]`` weight."""

    n, k = weights.shape
    if packed is None:
        packed = torch.empty((n, k // 2), device=weights.device, dtype=torch.uint8)
    if sfb is None:
        sfb = torch.empty((_swizzled_sf_bytes(n, k),), device=weights.device, dtype=torch.uint8)
    ops.quantize_weights_nvfp4_bf16(weights, packed, sfb)
    return packed, sfb


def grouped_w4a4_gemv_bf16(
    activations_packed: torch.Tensor,
    weight_stack: torch.Tensor,
    sfa: torch.Tensor,
    sfb_stack: torch.Tensor,
    alpha_stack: torch.Tensor,
    expert_idx: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute all token/top-k W4A4 expert projections in one launch.

    Inputs use token-major routing: ``expert_idx[M,top_k]`` and output
    ``[M,top_k,N]``. The device index tensor is read on every graph replay.
    """

    m = activations_packed.shape[0]
    top_k = expert_idx.shape[1]
    n = weight_stack.shape[1]
    if out is None:
        out = torch.empty((m, top_k, n), device=activations_packed.device, dtype=torch.bfloat16)
    ops.grouped_w4a4_gemv_bf16(
        activations_packed, weight_stack, sfa, sfb_stack, alpha_stack, expert_idx, out
    )
    return out


def grouped_w4a4_gemv_from_bf16(
    activations: torch.Tensor,
    weight_stack: torch.Tensor,
    sfb_stack: torch.Tensor,
    alpha_stack: torch.Tensor,
    expert_idx: torch.Tensor,
    *,
    packed: Optional[torch.Tensor] = None,
    sfa: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quantize ``[M,K]`` once, then launch all ``M*top_k`` projections.

    Supplying all three buffers makes this two-launch composition allocation
    free and CUDA Graph replay safe.
    """

    packed, sfa = quantize_activations_nvfp4_bf16(
        activations, packed=packed, sfa=sfa
    )
    return grouped_w4a4_gemv_bf16(
        packed, weight_stack, sfa, sfb_stack, alpha_stack, expert_idx, out=out
    )


__all__ = [
    "grouped_w4a4_gemv_bf16",
    "grouped_w4a4_gemv_from_bf16",
    "grouped_w4a16_gemv_bf16",
    "quantize_activations_nvfp4_bf16",
    "quantize_weights_nvfp4_bf16",
    "w4a16_decode_gemv_bf16",
]
