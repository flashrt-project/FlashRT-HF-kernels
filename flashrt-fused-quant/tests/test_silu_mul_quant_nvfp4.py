import math
import struct

import pytest
import torch

from flashrt_fused_quant import (
    nvfp4_swizzled_scale_bytes,
    silu_mul_merged_quant_nvfp4_swizzled_bf16,
    silu_mul_quant_nvfp4_swizzled_bf16,
)


def _float_to_fp4_e2m1(v: float) -> int:
    sign = 0x8 if v < 0 else 0x0
    a = abs(v)
    if a < 0.25:
        mag = 0
    elif a < 0.75:
        mag = 1
    elif a < 1.25:
        mag = 2
    elif a < 1.75:
        mag = 3
    elif a < 2.5:
        mag = 4
    elif a < 3.5:
        mag = 5
    elif a < 5.0:
        mag = 6
    else:
        mag = 7
    return sign | mag


def _float_to_ue4m3_ceil(v: float) -> int:
    if v <= 0:
        return 0
    if v > 240:
        return 0xFE
    bits = struct.unpack("I", struct.pack("f", float(v)))[0]
    float_exp = ((bits >> 23) & 0xFF) - 127
    frac = bits & 0x7FFFFF
    ue_exp = float_exp + 7
    if ue_exp <= 0:
        m = math.ceil(v * 512.0)
        if m > 7:
            return (1 << 3) | 0
        if m < 1:
            m = 1
        return m
    if ue_exp >= 15:
        return 0xFE
    m = frac >> 20
    if frac & 0xFFFFF:
        m += 1
    if m >= 8:
        m = 0
        ue_exp += 1
    if ue_exp >= 15:
        return 0xFE
    return (ue_exp << 3) | m


def _ue4m3_to_float(byte: int) -> float:
    e = (byte >> 3) & 0xF
    m = byte & 0x7
    if e == 0:
        return math.ldexp(m / 8.0, -6)
    return math.ldexp(1.0 + m / 8.0, e - 7)


def _reference(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = gate.shape
    gate_cpu = gate.cpu()
    up_cpu = up.cpu()
    vals = torch.empty((rows, cols), dtype=torch.bfloat16)
    for row in range(rows):
        for col in range(cols):
            g = float(gate_cpu[row, col])
            u = float(up_cpu[row, col])
            silu = g / (1.0 + math.exp(-g))
            silu_bf = float(torch.tensor(silu, dtype=torch.bfloat16))
            vals[row, col] = torch.tensor(silu_bf * u, dtype=torch.bfloat16)

    packed = torch.empty((rows, cols // 2), dtype=torch.uint8)
    scale_linear = torch.empty((rows, cols // 16), dtype=torch.uint8)
    for row in range(rows):
        for block in range(cols // 16):
            block_vals = vals[row, block * 16 : (block + 1) * 16].float()
            amax = float(block_vals.abs().max())
            scale_byte = _float_to_ue4m3_ceil(amax / 6.0)
            scale = _ue4m3_to_float(scale_byte)
            inv_scale = 1.0 / scale if scale > 0 else 0.0
            scale_linear[row, block] = scale_byte
            for pair in range(8):
                i = block * 16 + pair * 2
                lo = _float_to_fp4_e2m1(float(vals[row, i]) * inv_scale)
                hi = _float_to_fp4_e2m1(float(vals[row, i + 1]) * inv_scale)
                packed[row, i // 2] = (hi << 4) | lo

    scales = _swizzle(scale_linear)
    return packed, scales


def _swizzle(scales: torch.Tensor) -> torch.Tensor:
    rows, n_blocks = scales.shape
    n_col_super = (n_blocks + 3) // 4
    out = torch.zeros(
        (nvfp4_swizzled_scale_bytes(rows, n_blocks * 16),),
        dtype=torch.uint8,
    )
    for row in range(rows):
        rb = row // 128
        ri = row % 128
        for block in range(n_blocks):
            cb = block // 4
            ci = block % 4
            super_idx = rb * n_col_super + cb
            inner_off = (ri % 32) * 16 + (ri // 32) * 4 + ci
            out[super_idx * 512 + inner_off] = scales[row, block]
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("rows", "cols"), [(1, 16), (3, 64), (33, 128)])
def test_silu_mul_quant_nvfp4_swizzled_bf16(rows, cols):
    torch.manual_seed(0)
    gate = (torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
    up = (torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()

    packed, scales = silu_mul_quant_nvfp4_swizzled_bf16(gate, up)
    expected_packed, expected_scales = _reference(gate, up)

    torch.testing.assert_close(packed.cpu(), expected_packed)
    torch.testing.assert_close(scales.cpu(), expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_mul_merged_quant_nvfp4_swizzled_bf16():
    torch.manual_seed(1)
    rows, cols = 4, 64
    gate = (torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
    up = (torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
    merged = torch.cat([gate, up], dim=1).contiguous()

    packed, scales = silu_mul_merged_quant_nvfp4_swizzled_bf16(merged)
    expected_packed, expected_scales = _reference(gate, up)

    torch.testing.assert_close(packed.cpu(), expected_packed)
    torch.testing.assert_close(scales.cpu(), expected_scales)


def test_nvfp4_swizzled_scale_bytes_rejects_invalid_shape():
    with pytest.raises(ValueError, match="rows"):
        nvfp4_swizzled_scale_bytes(0, 64)
    with pytest.raises(ValueError, match="cols"):
        nvfp4_swizzled_scale_bytes(1, 15)
