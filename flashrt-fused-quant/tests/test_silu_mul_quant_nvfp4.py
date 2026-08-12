#!/usr/bin/env python3
"""Correctness tests for flashrt-fused-quant (SiLU*up fused into NVFP4)."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import struct
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-fused-quant"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 12:
        return "12.0a"
    if (major, minor) == (11, 0):
        return "11.0a"
    return f"{major}.{minor}"


def _swizzled_bytes(rows: int, cols: int) -> int:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if cols <= 0 or cols % 16 != 0:
        raise ValueError("cols must be positive and divisible by 16")
    n_blocks = cols // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 512


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def scale_bytes(self, rows: int, cols: int) -> int:
        return _swizzled_bytes(rows, cols)

    def silu_mul_quant(self, gate, up, *, packed=None, scales=None):
        rows, cols = gate.shape
        if packed is None:
            packed = torch.empty((rows, cols // 2), device=gate.device, dtype=torch.uint8)
        if scales is None:
            scales = torch.zeros(_swizzled_bytes(rows, cols), device=gate.device, dtype=torch.uint8)
        self._ops.silu_mul_quant_nvfp4_swizzled_bf16(gate, up, packed, scales)
        return packed, scales

    def silu_mul_merged_quant(self, merged, *, packed=None, scales=None):
        rows, merged_cols = merged.shape
        cols = merged_cols // 2
        if packed is None:
            packed = torch.empty((rows, cols // 2), device=merged.device, dtype=torch.uint8)
        if scales is None:
            scales = torch.zeros(_swizzled_bytes(rows, cols), device=merged.device, dtype=torch.uint8)
        self._ops.silu_mul_merged_quant_nvfp4_swizzled_bf16(merged, packed, scales)
        return packed, scales


class InstalledOps:
    def __init__(self, module) -> None:
        self._module = module

    def scale_bytes(self, rows: int, cols: int) -> int:
        return int(self._module.nvfp4_swizzled_scale_bytes(rows, cols))

    def silu_mul_quant(self, gate, up, *, packed=None, scales=None):
        return self._module.silu_mul_quant_nvfp4_swizzled_bf16(gate, up, packed=packed, scales=scales)

    def silu_mul_merged_quant(self, merged, *, packed=None, scales=None):
        return self._module.silu_mul_merged_quant_nvfp4_swizzled_bf16(merged, packed=packed, scales=scales)


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "flashrt_fused_quant_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "silu_mul_to_nvfp4_swizzled.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return InstalledOps(importlib.import_module("flashrt_fused_quant"))
    finally:
        if artifact:
            sys.path.remove(artifact)


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


def _swizzle(scales: torch.Tensor) -> torch.Tensor:
    rows, n_blocks = scales.shape
    n_col_super = (n_blocks + 3) // 4
    out = torch.zeros(_swizzled_bytes(rows, n_blocks * 16), dtype=torch.uint8)
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
    return packed, _swizzle(scale_linear)


def _check_silu_mul_quant(ops, rows: int, cols: int) -> None:
    torch.manual_seed(0)
    gate = (torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
    up = (torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
    packed, scales = ops.silu_mul_quant(gate, up)
    exp_packed, exp_scales = _reference(gate, up)
    torch.testing.assert_close(packed.cpu(), exp_packed)
    torch.testing.assert_close(scales.cpu(), exp_scales)
    print(f"PASS silu_mul_quant rows={rows} cols={cols}: exact")


def _check_merged_quant(ops) -> None:
    torch.manual_seed(1)
    rows, cols = 4, 64
    gate = (torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
    up = (torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
    merged = torch.cat([gate, up], dim=1).contiguous()
    packed, scales = ops.silu_mul_merged_quant(merged)
    exp_packed, exp_scales = _reference(gate, up)
    torch.testing.assert_close(packed.cpu(), exp_packed)
    torch.testing.assert_close(scales.cpu(), exp_scales)
    print("PASS silu_mul_merged_quant: exact")


def _check_scale_bytes(ops) -> None:
    for fn_args, match in [((0, 64), "rows"), ((1, 15), "cols")]:
        try:
            ops.scale_bytes(*fn_args)
        except (ValueError, RuntimeError) as exc:
            if match not in str(exc):
                raise AssertionError(f"expected error containing {match!r}, got {exc}")
        else:
            raise AssertionError(f"expected error containing {match!r}")
    print("PASS scale_bytes rejects invalid shape")


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    cases = [(1, 16), (3, 64)] if args.mode == "smoke" else [(1, 16), (3, 64), (33, 128)]
    for rows, cols in cases:
        _check_silu_mul_quant(ops, rows, cols)
    _check_merged_quant(ops)
    _check_scale_bytes(ops)
    print(f"PASS flashrt-fused-quant {args.backend} mode={args.mode}: "
          f"{len(cases) + 2} checks")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"passed": 1, "total": 1, "backend": args.backend}) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
