# SPDX-License-Identifier: Apache-2.0
"""NVFP4 helper API compatible with MiniMaxAI/msa.

The quantization entrypoints use Transformer Engine when it is installed.  The
layout helpers and reference dequantizer are pure PyTorch and are included so
callers can prepare and validate NVFP4 metadata with the same public API names
as the upstream MiniMaxAI package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


NVFP4_BLOCK_SIZE = 16
NVFP4_FP4_MAX = 6.0
NVFP4_FP8_E4M3_MAX = 448.0


@dataclass(frozen=True)
class Nvfp4QuantizedTensor:
    data: torch.Tensor
    scale_128x4: torch.Tensor
    global_scale: torch.Tensor
    logical_scale_shape: Tuple[int, int]
    original_shape: Tuple[int, ...]


def _round_up(x: int, multiple: int) -> int:
    return ((int(x) + multiple - 1) // multiple) * multiple


def nvfp4_scale_128x4_offset(
    row: torch.Tensor,
    col: torch.Tensor,
    scale_cols: int,
) -> torch.Tensor:
    tiles_n = _round_up(scale_cols, 4) // 4
    tile_m = row // 128
    tile_n = col // 4
    outer = row % 128
    inner = col % 4
    return (
        (tile_m * tiles_n + tile_n) * 512
        + (outer % 32) * 16
        + (outer // 32) * 4
        + inner
    )


def swizzle_nvfp4_scale_to_128x4(
    scale: torch.Tensor,
    *,
    rows: int,
    cols: int,
) -> torch.Tensor:
    if scale.ndim != 2:
        raise ValueError(f"scale must be 2D, got shape {tuple(scale.shape)}")

    rows = int(rows)
    cols = int(cols)
    padded_rows = _round_up(rows, 128)
    padded_cols = _round_up(cols, 4)
    if scale.shape[0] < rows or scale.shape[1] < cols:
        raise ValueError(
            "scale is smaller than the requested logical shape: "
            f"got {tuple(scale.shape)}, need at least {(rows, cols)}"
        )

    logical = scale[:rows, :cols].contiguous()
    if logical.shape != (padded_rows, padded_cols):
        logical = torch.nn.functional.pad(
            logical.to(torch.float32),
            (0, padded_cols - cols, 0, padded_rows - rows),
        ).to(scale.dtype)
    swizzled = torch.empty_like(logical)

    row = torch.arange(padded_rows, device=scale.device, dtype=torch.int64)[:, None]
    col = torch.arange(padded_cols, device=scale.device, dtype=torch.int64)[None, :]
    offset = nvfp4_scale_128x4_offset(row, col, padded_cols).reshape(-1)
    swizzled.reshape(-1)[offset] = logical.reshape(-1)
    return swizzled


def nvfp4_global_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    return amax.to(torch.float32) / (NVFP4_FP8_E4M3_MAX * NVFP4_FP4_MAX)


def _import_te_nvfp4_quantizer():
    try:
        from transformer_engine.pytorch.tensor import NVFP4Quantizer
    except Exception as exc:
        raise RuntimeError(
            "Transformer Engine NVFP4 quantization is unavailable. Install a "
            "Transformer Engine build with NVFP4 PyTorch support to use "
            "quantize_bf16_to_nvfp4_128x4."
        ) from exc
    return NVFP4Quantizer


def quantize_bf16_to_nvfp4_128x4(x: torch.Tensor) -> Nvfp4QuantizedTensor:
    if not x.is_cuda:
        raise ValueError("NVFP4 quantization requires a CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"x must be bf16 or fp16, got {x.dtype}")
    if x.ndim < 2:
        raise ValueError(f"x must have at least 2 dimensions, got {x.ndim}")
    if x.shape[-1] % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"last dimension must be divisible by {NVFP4_BLOCK_SIZE}, got {x.shape[-1]}"
        )

    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    if rows % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            "flattened row dimension must be divisible by "
            f"{NVFP4_BLOCK_SIZE}, got {rows}"
        )

    NVFP4Quantizer = _import_te_nvfp4_quantizer()
    quantizer = NVFP4Quantizer(rowwise=True, columnwise=False)
    qx = quantizer.quantize(x.contiguous())
    meta = qx.get_metadata()

    data = meta["rowwise_data"]
    if data.dtype is not torch.uint8:
        data = data.view(torch.uint8)
    logical_scale = meta["rowwise_scale_inv"]
    amax = meta["amax_rowwise"]
    scale_cols = int(x.shape[-1]) // NVFP4_BLOCK_SIZE
    scale_128x4 = swizzle_nvfp4_scale_to_128x4(
        logical_scale,
        rows=rows,
        cols=scale_cols,
    )
    global_scale = nvfp4_global_scale_from_amax(amax).contiguous()

    return Nvfp4QuantizedTensor(
        data=data,
        scale_128x4=scale_128x4,
        global_scale=global_scale,
        logical_scale_shape=(rows, scale_cols),
        original_shape=tuple(int(v) for v in x.shape),
    )


def quantize_kv_bf16_to_nvfp4_128x4(
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[Nvfp4QuantizedTensor, Nvfp4QuantizedTensor]:
    return quantize_bf16_to_nvfp4_128x4(k), quantize_bf16_to_nvfp4_128x4(v)


def dequantize_nvfp4_128x4_to_bf16(
    qx: Nvfp4QuantizedTensor,
    *,
    include_global_scale: bool = True,
) -> torch.Tensor:
    data = qx.data if qx.data.dtype is torch.uint8 else qx.data.view(torch.uint8)
    if data.shape[-1] * 2 != qx.original_shape[-1]:
        raise ValueError(
            "packed data last dimension does not match original shape: "
            f"{data.shape[-1]} packed vs {qx.original_shape[-1]} logical"
        )

    rows, scale_cols = qx.logical_scale_shape
    logical_dim = int(qx.original_shape[-1])
    if scale_cols * NVFP4_BLOCK_SIZE != logical_dim:
        raise ValueError(
            "logical scale columns do not match original last dimension: "
            f"{scale_cols} scale cols vs dim {logical_dim}"
        )

    fp4_lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=data.device,
    )
    packed = data.reshape(rows, logical_dim // 2)
    lo = packed & 0x0F
    hi = packed >> 4
    values = torch.empty((rows, logical_dim), dtype=torch.float32, device=data.device)
    values[:, 0::2] = fp4_lut[lo.long()]
    values[:, 1::2] = fp4_lut[hi.long()]

    row = torch.arange(rows, device=data.device, dtype=torch.int64)[:, None]
    col = torch.arange(scale_cols, device=data.device, dtype=torch.int64)[None, :]
    offset = nvfp4_scale_128x4_offset(row, col, scale_cols)
    scale_u8 = qx.scale_128x4.reshape(-1)[offset.reshape(-1)].reshape(rows, scale_cols)
    scale = scale_u8.view(torch.float8_e4m3fn).to(torch.float32)
    scale = scale.repeat_interleave(NVFP4_BLOCK_SIZE, dim=1)
    out = values * scale
    if include_global_scale:
        out = out * qx.global_scale.reshape(-1)[0].to(torch.float32)
    return out.reshape(qx.original_shape).to(torch.bfloat16)


__all__ = [
    "Nvfp4QuantizedTensor",
    "quantize_bf16_to_nvfp4_128x4",
    "quantize_kv_bf16_to_nvfp4_128x4",
    "dequantize_nvfp4_128x4_to_bf16",
    "swizzle_nvfp4_scale_to_128x4",
    "nvfp4_global_scale_from_amax",
]
