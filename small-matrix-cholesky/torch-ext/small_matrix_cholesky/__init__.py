"""Batched FP32 Cholesky kernels for small CUDA matrices."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(
    add_op_namespace_prefix("cholesky_small_fp32_out")
)
def _cholesky_small_fp32_out_fake(
    input: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if input.dim() < 2:
        raise RuntimeError("input must have at least two dimensions")
    if input.shape != output.shape:
        raise RuntimeError("output must have the same shape as input")
    if input.shape[-2] != input.shape[-1]:
        raise RuntimeError("the last two dimensions must be square")
    if input.shape[-1] not in (32, 64, 128):
        raise RuntimeError("supported matrix orders are 32, 64, and 128")
    return None


def cholesky_small_fp32(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute a lower-triangular Cholesky factor for small FP32 matrices.

    ``input`` must be a contiguous CUDA FP32 tensor whose last two dimensions
    are ``(n, n)`` with ``n`` equal to 32, 64, or 128. All leading dimensions
    are flattened into a batch. The input matrices must be symmetric positive
    definite. The output's upper triangle is explicitly zero.
    """

    if out is None:
        out = torch.empty_like(input)
    ops.cholesky_small_fp32_out(input, out)
    return out


__all__ = ["cholesky_small_fp32"]
