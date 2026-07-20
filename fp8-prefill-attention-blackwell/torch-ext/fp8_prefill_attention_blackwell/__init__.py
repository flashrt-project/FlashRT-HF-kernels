"""FP8 causal GQA prefill attention for Blackwell GPUs."""

from __future__ import annotations

import math
import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.custom_op(
    add_op_namespace_prefix("_fp8_causal_gqa_attention_bf16_out"),
    mutates_args=(),
    device_types="cuda",
)
def _compileable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    output = torch.empty_like(query, dtype=torch.bfloat16)
    ops.fp8_causal_gqa_attention_bf16_out(query, key, value, softmax_scale, output)
    return output


@torch.library.register_fake(
    add_op_namespace_prefix("_fp8_causal_gqa_attention_bf16_out")
)
def _fake(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    if query.ndim != 3 or query.shape[1:] != (32, 128):
        raise RuntimeError("query must have shape (sequence, 32, 128)")
    if key.shape != value.shape or key.shape != (query.shape[0], 8, 128):
        raise RuntimeError("key/value must have shape (sequence, 8, 128)")
    if query.shape[0] < 256 or query.shape[0] % 128:
        raise RuntimeError("sequence length must be a multiple of 128 and at least 256")
    return torch.empty_like(query, dtype=torch.bfloat16)


def fp8_causal_gqa_attention_bf16(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run causal GQA self-attention for the documented fixed head layout."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(128)
    if out is None:
        return _compileable(query, key, value, float(softmax_scale))
    ops.fp8_causal_gqa_attention_bf16_out(query, key, value, float(softmax_scale), out)
    return out


__all__ = ["fp8_causal_gqa_attention_bf16"]
