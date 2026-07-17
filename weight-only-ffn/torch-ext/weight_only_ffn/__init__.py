"""Weight-only BF16-activation linear and FFN regions."""

from __future__ import annotations

from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


def w4_sfb_size_bytes(rows: int, cols: int) -> int:
    if rows <= 0 or cols <= 0 or cols % 64:
        raise ValueError("rows must be positive and cols must be divisible by 64")
    return ((int(rows) + 127) // 128) * (((int(cols) // 16) + 3) // 4) * 512


@torch.library.register_fake(add_op_namespace_prefix("quantize_w4_weight_bf16"))
def _quantize_w4_fake(weight: torch.Tensor, packed: torch.Tensor, sfb: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("dequantize_w4_weight_bf16"))
def _dequantize_w4_fake(packed: torch.Tensor, sfb: torch.Tensor, weight: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("w4a16_linear_bf16"))
def _w4_linear_fake(
    x: torch.Tensor,
    packed: torch.Tensor,
    sfb: torch.Tensor,
    alpha: float,
    variant: int,
    out: torch.Tensor,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_w8_weight_bf16"))
def _quantize_w8_fake(weight: torch.Tensor, quantized: torch.Tensor, scales: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("dequantize_w8_weight_bf16"))
def _dequantize_w8_fake(quantized: torch.Tensor, scales: torch.Tensor, weight: torch.Tensor) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("w8a16_linear_bf16"))
def _w8_linear_fake(
    x: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    variant: int,
    out: torch.Tensor,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("w4a16_gated_ffn_bf16"))
def _w4_gated_fake(
    x, gu_w, gu_s, dn_w, dn_s, gu_b, dn_b, gelu, gu_alpha, dn_alpha,
    variant, gu, hidden, out,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("w8a16_gated_ffn_bf16"))
def _w8_gated_fake(
    x, gu_w, gu_s, dn_w, dn_s, gu_b, dn_b, gelu, variant, gu, hidden, out,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("w4a16_gelu_ffn_bf16"))
def _w4_gelu_fake(
    x, up_w, up_s, dn_w, dn_s, up_b, dn_b, up_alpha, dn_alpha,
    variant, up, hidden, out,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("w8a16_gelu_ffn_bf16"))
def _w8_gelu_fake(
    x, up_w, up_s, dn_w, dn_s, up_b, dn_b, variant, up, hidden, out,
) -> None:
    return None


def quantize_w4_weight_bf16(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a static BF16 weight matrix into NVFP4 data and SFB scales."""
    n, k = weight.shape
    packed = torch.empty((n, k // 2), device=weight.device, dtype=torch.uint8)
    sfb = torch.empty((w4_sfb_size_bytes(n, k),), device=weight.device, dtype=torch.uint8)
    ops.quantize_w4_weight_bf16(weight, packed, sfb)
    return packed, sfb


def dequantize_w4_weight_bf16(
    packed: torch.Tensor, sfb: torch.Tensor, *, cols: int
) -> torch.Tensor:
    weight = torch.empty((packed.shape[0], int(cols)), device=packed.device, dtype=torch.bfloat16)
    ops.dequantize_w4_weight_bf16(packed, sfb, weight)
    return weight


def quantize_w8_weight_bf16(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize a static BF16 weight matrix per output channel."""
    quantized = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty((weight.shape[0],), device=weight.device, dtype=torch.float32)
    ops.quantize_w8_weight_bf16(weight, quantized, scales)
    return quantized, scales


def dequantize_w8_weight_bf16(
    quantized: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    weight = torch.empty_like(quantized, dtype=torch.bfloat16)
    ops.dequantize_w8_weight_bf16(quantized, scales, weight)
    return weight


def w4a16_linear_bf16(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_sfb: torch.Tensor,
    *,
    alpha: float = 1.0,
    variant: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((x.shape[0], weight_packed.shape[0]), device=x.device, dtype=torch.bfloat16)
    ops.w4a16_linear_bf16(x, weight_packed, weight_sfb, float(alpha), int(variant), out)
    return out


def w8a16_linear_bf16(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    variant: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty((x.shape[0], weight_int8.shape[0]), device=x.device, dtype=torch.bfloat16)
    ops.w8a16_linear_bf16(x, weight_int8, weight_scale, int(variant), out)
    return out


def _gated_workspace(x: torch.Tensor, gate_up_rows: int, out_rows: int, workspace, out):
    hidden_size = gate_up_rows // 2
    if workspace is None:
        gate_up = torch.empty((x.shape[0], gate_up_rows), device=x.device, dtype=torch.bfloat16)
        hidden = torch.empty((x.shape[0], hidden_size), device=x.device, dtype=torch.bfloat16)
    else:
        gate_up, hidden = workspace
    if out is None:
        out = torch.empty((x.shape[0], out_rows), device=x.device, dtype=torch.bfloat16)
    return gate_up, hidden, out


def _gelu_workspace(x: torch.Tensor, hidden_size: int, out_rows: int, workspace, out):
    if workspace is None:
        up = torch.empty((x.shape[0], hidden_size), device=x.device, dtype=torch.bfloat16)
        hidden = torch.empty_like(up)
    else:
        up, hidden = workspace
    if out is None:
        out = torch.empty((x.shape[0], out_rows), device=x.device, dtype=torch.bfloat16)
    return up, hidden, out


def _w4_gated(
    x, gate_up_weight, gate_up_sfb, down_weight, down_sfb, *, gelu,
    gate_up_bias=None, down_bias=None, gate_up_alpha=1.0, down_alpha=1.0,
    variant=0, workspace=None, out=None,
):
    gate_up, hidden, out = _gated_workspace(
        x, gate_up_weight.shape[0], down_weight.shape[0], workspace, out
    )
    ops.w4a16_gated_ffn_bf16(
        x, gate_up_weight, gate_up_sfb, down_weight, down_sfb,
        gate_up_bias, down_bias, bool(gelu), float(gate_up_alpha),
        float(down_alpha), int(variant), gate_up, hidden, out,
    )
    return out


def w4a16_swiglu_ffn_bf16(x, gate_up_weight, gate_up_sfb, down_weight, down_sfb, **kwargs):
    return _w4_gated(x, gate_up_weight, gate_up_sfb, down_weight, down_sfb, gelu=False, **kwargs)


def w4a16_geglu_ffn_bf16(x, gate_up_weight, gate_up_sfb, down_weight, down_sfb, **kwargs):
    return _w4_gated(x, gate_up_weight, gate_up_sfb, down_weight, down_sfb, gelu=True, **kwargs)


def _w8_gated(
    x, gate_up_weight, gate_up_scale, down_weight, down_scale, *, gelu,
    gate_up_bias=None, down_bias=None, variant=0, workspace=None, out=None,
):
    gate_up, hidden, out = _gated_workspace(
        x, gate_up_weight.shape[0], down_weight.shape[0], workspace, out
    )
    ops.w8a16_gated_ffn_bf16(
        x, gate_up_weight, gate_up_scale, down_weight, down_scale,
        gate_up_bias, down_bias, bool(gelu), int(variant), gate_up, hidden, out,
    )
    return out


def w8a16_swiglu_ffn_bf16(x, gate_up_weight, gate_up_scale, down_weight, down_scale, **kwargs):
    return _w8_gated(x, gate_up_weight, gate_up_scale, down_weight, down_scale, gelu=False, **kwargs)


def w8a16_geglu_ffn_bf16(x, gate_up_weight, gate_up_scale, down_weight, down_scale, **kwargs):
    return _w8_gated(x, gate_up_weight, gate_up_scale, down_weight, down_scale, gelu=True, **kwargs)


def w4a16_gelu_ffn_bf16(
    x, up_weight, up_sfb, down_weight, down_sfb, *, up_bias=None,
    down_bias=None, up_alpha=1.0, down_alpha=1.0, variant=0,
    workspace=None, out=None,
):
    up, hidden, out = _gelu_workspace(x, up_weight.shape[0], down_weight.shape[0], workspace, out)
    ops.w4a16_gelu_ffn_bf16(
        x, up_weight, up_sfb, down_weight, down_sfb, up_bias, down_bias,
        float(up_alpha), float(down_alpha), int(variant), up, hidden, out,
    )
    return out


def w8a16_gelu_ffn_bf16(
    x, up_weight, up_scale, down_weight, down_scale, *, up_bias=None,
    down_bias=None, variant=0, workspace=None, out=None,
):
    up, hidden, out = _gelu_workspace(x, up_weight.shape[0], down_weight.shape[0], workspace, out)
    ops.w8a16_gelu_ffn_bf16(
        x, up_weight, up_scale, down_weight, down_scale, up_bias, down_bias,
        int(variant), up, hidden, out,
    )
    return out


__all__ = [
    "dequantize_w4_weight_bf16",
    "dequantize_w8_weight_bf16",
    "quantize_w4_weight_bf16",
    "quantize_w8_weight_bf16",
    "w4_sfb_size_bytes",
    "w4a16_geglu_ffn_bf16",
    "w4a16_gelu_ffn_bf16",
    "w4a16_linear_bf16",
    "w4a16_swiglu_ffn_bf16",
    "w8a16_geglu_ffn_bf16",
    "w8a16_gelu_ffn_bf16",
    "w8a16_linear_bf16",
    "w8a16_swiglu_ffn_bf16",
]
