"""Grouped NVFP4 MoE GEMM for Blackwell prefill workloads."""

from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix, ops


@torch.library.custom_op(
    add_op_namespace_prefix("_grouped_nvfp4_gemm_bf16"),
    mutates_args=(),
    device_types="cuda",
)
def _compileable(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    tile_expert: torch.Tensor,
    tile_rows: int,
    input_scale_stride: int,
    weight_stride: int,
    weight_scale_stride: int,
) -> torch.Tensor:
    output = torch.empty(
        (input.shape[0], weight.shape[1]), device=input.device, dtype=torch.bfloat16
    )
    ops.grouped_nvfp4_gemm_bf16_out(
        input,
        weight,
        input_scale,
        weight_scale,
        alpha,
        tile_expert,
        tile_rows,
        input_scale_stride,
        weight_stride,
        weight_scale_stride,
        output,
    )
    return output


@torch.library.register_fake(add_op_namespace_prefix("_grouped_nvfp4_gemm_bf16"))
def _fake(
    input,
    weight,
    input_scale,
    weight_scale,
    alpha,
    tile_expert,
    tile_rows,
    input_scale_stride,
    weight_stride,
    weight_scale_stride,
):
    if input.ndim != 2 or weight.ndim != 3:
        raise RuntimeError("invalid grouped GEMM tensor shapes")
    return torch.empty(
        (input.shape[0], weight.shape[1]), device=input.device, dtype=torch.bfloat16
    )


def grouped_nvfp4_gemm_bf16(
    input,
    weight,
    input_scale,
    weight_scale,
    alpha,
    tile_expert,
    *,
    tile_rows,
    input_scale_stride=0,
    weight_stride=None,
    weight_scale_stride=None,
    out=None,
):
    """Compute expert-selected packed NVFP4 GEMM tiles with BF16 output."""
    if weight_stride is None:
        weight_stride = weight[0].numel()
    if weight_scale_stride is None:
        weight_scale_stride = weight_scale[0].numel()
    if out is None:
        return _compileable(
            input,
            weight,
            input_scale,
            weight_scale,
            alpha,
            tile_expert,
            int(tile_rows),
            int(input_scale_stride),
            int(weight_stride),
            int(weight_scale_stride),
        )
    ops.grouped_nvfp4_gemm_bf16_out(
        input,
        weight,
        input_scale,
        weight_scale,
        alpha,
        tile_expert,
        int(tile_rows),
        int(input_scale_stride),
        int(weight_stride),
        int(weight_scale_stride),
        out,
    )
    return out


__all__ = ["grouped_nvfp4_gemm_bf16"]
