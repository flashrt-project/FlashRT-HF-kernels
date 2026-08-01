"""Blackwell FP16 fused MLP megakernels from FlashRT."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def _check(input, gate_weight, up_weight, gate_scratch, output) -> None:
    if input.dim() != 2 or gate_weight.dim() != 2 or up_weight.dim() != 2:
        raise RuntimeError("input and weights must be rank-2 tensors")
    if gate_weight.shape != up_weight.shape:
        raise RuntimeError("gate_weight and up_weight must have the same [N,K] shape")
    if input.shape[1] != gate_weight.shape[1]:
        raise RuntimeError("input K must match weight K")
    expected = (input.shape[0], gate_weight.shape[0])
    if gate_scratch.shape != expected or output.shape != expected:
        raise RuntimeError("gate_scratch and output must have shape [M,N]")


@torch.library.register_fake(add_op_namespace_prefix("fp16_geglu_fused_out"))
def _fp16_geglu_fused_out_fake(
    input, gate_weight, up_weight, gate_scratch, output
) -> None:
    _check(input, gate_weight, up_weight, gate_scratch, output)
    return None


def fp16_geglu_fused(
    input: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    gate_scratch: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``GELU(input @ gate.T) * (input @ up.T)`` in one kernel.

    Pass preallocated ``gate_scratch`` and ``output`` for CUDA Graph capture
    and allocation-free hot paths.
    """
    shape = (input.shape[0], gate_weight.shape[0])
    if gate_scratch is None:
        gate_scratch = torch.empty(shape, device=input.device, dtype=torch.float16)
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=torch.float16)
    ops.fp16_geglu_fused_out(
        input, gate_weight, up_weight, gate_scratch, output
    )
    return output


__all__ = ["fp16_geglu_fused"]
