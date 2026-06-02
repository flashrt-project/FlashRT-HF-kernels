"""FlashRT fused activation and quantization kernels."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import ops


def nvfp4_swizzled_scale_bytes(rows: int, cols: int) -> int:
    """Return byte count for a CUTLASS Sm1xx NVFP4 swizzled scale buffer."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if cols <= 0 or cols % 16 != 0:
        raise ValueError("cols must be positive and divisible by 16")
    n_blocks = cols // 16
    return ((rows + 127) // 128) * ((n_blocks + 3) // 4) * 512


def _allocate_outputs(
    rows: int,
    cols: int,
    device: torch.device,
    packed: Optional[torch.Tensor],
    scales: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if packed is None:
        packed = torch.empty((rows, cols // 2), device=device, dtype=torch.uint8)
    if scales is None:
        scales = torch.zeros(
            (nvfp4_swizzled_scale_bytes(rows, cols),),
            device=device,
            dtype=torch.uint8,
        )
    return packed, scales


def silu_mul_quant_nvfp4_swizzled_bf16(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    packed: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``SiLU(gate) * up`` and quantize to NVFP4 swizzled layout.

    ``gate`` and ``up`` must be contiguous BF16 tensors with shape
    ``(rows, cols)``. ``cols`` must be divisible by 16. The returned ``packed``
    tensor has shape ``(rows, cols / 2)`` and dtype ``torch.uint8``. The
    returned ``scales`` tensor is a flat CUTLASS Sm1xx swizzled UE4M3
    scale-factor byte buffer.
    """

    if gate.dim() != 2:
        raise ValueError("gate must have shape (rows, cols)")
    rows, cols = gate.shape
    packed, scales = _allocate_outputs(rows, cols, gate.device, packed, scales)
    ops.silu_mul_quant_nvfp4_swizzled_bf16(gate, up, packed, scales)
    return packed, scales


def silu_mul_merged_quant_nvfp4_swizzled_bf16(
    merged_gate_up: torch.Tensor,
    *,
    packed: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute merged ``[gate | up]`` SiLU product and NVFP4 quantization.

    ``merged_gate_up`` must be contiguous BF16 with shape ``(rows, 2 * cols)``
    and row layout ``[gate | up]``.
    """

    if merged_gate_up.dim() != 2:
        raise ValueError("merged_gate_up must have shape (rows, 2 * cols)")
    rows, merged_cols = merged_gate_up.shape
    if merged_cols % 2 != 0:
        raise ValueError("merged_gate_up.shape[1] must be even")
    cols = merged_cols // 2
    packed, scales = _allocate_outputs(
        rows, cols, merged_gate_up.device, packed, scales
    )
    ops.silu_mul_merged_quant_nvfp4_swizzled_bf16(
        merged_gate_up,
        packed,
        scales,
    )
    return packed, scales


__all__ = [
    "nvfp4_swizzled_scale_bytes",
    "silu_mul_quant_nvfp4_swizzled_bf16",
    "silu_mul_merged_quant_nvfp4_swizzled_bf16",
]
