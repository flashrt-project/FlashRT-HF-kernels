"""FP8 GQA self/cross-attention for SM100-family Blackwell GPUs."""

from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix, ops


@torch.library.register_fake(
    add_op_namespace_prefix("fp8_gqa_cross_attention_bf16_out")
)
def _out_fake(query, key, value, query_scale, key_scale, value_scale,
              output, lse, workspace) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise RuntimeError("query/key/value must have [B,S,H,128] layout")
    if query.shape[0] != key.shape[0] or query.shape[-1] != 128 or key.shape[-1] != 128:
        raise RuntimeError("batch must match and head dimension must be 128")
    if query.shape[2] % key.shape[2]:
        raise RuntimeError("query heads must be divisible by KV heads")
    if output.shape != query.shape:
        raise RuntimeError("output must have the query shape")
    return None


@torch.library.custom_op(
    add_op_namespace_prefix("_fp8_gqa_cross_attention_bf16"),
    mutates_args=(),
    device_types="cuda",
)
def _compiled(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              query_scale: float, key_scale: float,
              value_scale: float) -> torch.Tensor:
    output = torch.empty_like(query, dtype=torch.bfloat16)
    rounded_sq = (query.shape[1] + 127) // 128 * 128
    lse = torch.empty(
        (query.shape[0], query.shape[2], rounded_sq),
        device=query.device, dtype=torch.float32
    )
    workspace = torch.empty(4 * 1024 * 1024, device=query.device, dtype=torch.uint8)
    ops.fp8_gqa_cross_attention_bf16_out(
        query, key, value, query_scale, key_scale, value_scale,
        output, lse, workspace
    )
    return output


@torch.library.register_fake(
    add_op_namespace_prefix("_fp8_gqa_cross_attention_bf16")
)
def _compiled_fake(query, key, value, query_scale, key_scale, value_scale):
    return torch.empty_like(query, dtype=torch.bfloat16)


def fp8_gqa_cross_attention_bf16(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_scale: float,
    key_scale: float,
    value_scale: float,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run non-causal FP8 GQA attention and return BF16 output."""
    if output is None and lse is None and workspace is None:
        return _compiled(
            query, key, value, float(query_scale), float(key_scale),
            float(value_scale)
        )
    if output is None:
        output = torch.empty_like(query, dtype=torch.bfloat16)
    rounded_sq = (query.shape[1] + 127) // 128 * 128
    if lse is None:
        lse = torch.empty(
            (query.shape[0], query.shape[2], rounded_sq),
            device=query.device, dtype=torch.float32
        )
    if workspace is None:
        workspace = torch.empty(4 * 1024 * 1024, device=query.device, dtype=torch.uint8)
    ops.fp8_gqa_cross_attention_bf16_out(
        query, key, value, float(query_scale), float(key_scale),
        float(value_scale), output, lse, workspace
    )
    return output


__all__ = ["fp8_gqa_cross_attention_bf16"]
