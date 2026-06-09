#!/usr/bin/env python3
"""Accuracy-first sweep for the FlashRT v1 kernel batch.

Run this against installed or locally built packages before any release build
or benchmark table is accepted.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import json
import math
import os
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


FP8_QUANT_SHAPES = [
    ("decode_m1", 1, 4096),
    ("decode_m2", 2, 4096),
    ("decode_m4", 4, 4096),
    ("decode_m8", 8, 4096),
    ("small_m16", 16, 4096),
    ("small_m32", 32, 4096),
    ("prefill_m64", 64, 4096),
    ("prefill_m128", 128, 4096),
    ("prefill_m256", 256, 4096),
    ("wide_n8192_m16", 16, 8192),
    ("wide_n8192_m128", 128, 8192),
    ("vla_n12288_m16", 16, 12288),
    ("vla_n12288_m64", 64, 12288),
    ("vla_n16384_m16", 16, 16384),
    ("vla_n16384_m64", 64, 16384),
]

BF16_LINEAR_SHAPES = [
    ("pi05_action_in", 10, 32, 1024),
    ("pi05_qkv", 10, 1024, 2560),
    ("pi05_o_proj", 10, 2048, 1024),
    ("pi05_action_out", 10, 1024, 32),
    ("decode_m1_1024", 1, 1024, 1024),
    ("decode_m1_qkv", 1, 1024, 2560),
    ("decode_m8_1024", 8, 1024, 1024),
    ("decode_m8_qkv", 8, 1024, 2560),
    ("decode_m10_1024", 10, 1024, 1024),
    ("decode_m10_qkv", 10, 1024, 2560),
    ("decode_m16_1024", 16, 1024, 1024),
    ("decode_m16_qkv", 16, 1024, 2560),
    ("vlm_m512_square", 512, 1152, 1152),
    ("vlm_m512_wide", 512, 1152, 4304),
    ("vla_m1024_square", 1024, 2048, 2048),
    ("vla_m1024_wide", 1024, 2048, 8192),
]

FP8_FFN_SHAPES = [
    ("small", 16, 128, 256, 128),
    ("pi05_vision", 512, 1152, 4304, 1152),
    ("pi05_decoder", 10, 1024, 4096, 1024),
    ("groot_vit", 128, 1024, 4096, 1024),
    ("groot_deepstack", 128, 4096, 4096, 2048),
    ("groot_vl_self_attn_long", 2520, 2048, 8192, 2048),
    ("groot_action_dit", 41, 1536, 6144, 1536),
]

FUSED_QUANT_SHAPES = [
    ("decode_r1_h4096", 1, 4096),
    ("decode_r2_h4096", 2, 4096),
    ("decode_r4_h4096", 4, 4096),
    ("decode_r8_h4096", 8, 4096),
    ("decode_r1_h8192", 1, 8192),
    ("decode_r4_h8192", 4, 8192),
    ("decode_r8_h8192", 8, 8192),
    ("decode_r1_h12288", 1, 12288),
    ("decode_r4_h12288", 4, 12288),
    ("decode_r8_h12288", 8, 12288),
    ("decode_r1_h16384", 1, 16384),
    ("decode_r4_h16384", 4, 16384),
    ("decode_r8_h16384", 8, 16384),
    ("small_r16_h4096", 16, 4096),
    ("small_r32_h4096", 32, 4096),
    ("small_r16_h8192", 16, 8192),
    ("small_r32_h8192", 32, 8192),
    ("small_r16_h12288", 16, 12288),
    ("small_r32_h12288", 32, 12288),
    ("small_r16_h16384", 16, 16384),
    ("small_r32_h16384", 32, 16384),
    ("prefill_r64_h4096", 64, 4096),
    ("prefill_r128_h4096", 128, 4096),
    ("prefill_r256_h4096", 256, 4096),
    ("video_r1024_h4096", 1024, 4096),
    ("video_r2520_h4096", 2520, 4096),
    ("prefill_r64_h8192", 64, 8192),
    ("prefill_r128_h8192", 128, 8192),
    ("prefill_r256_h8192", 256, 8192),
    ("video_r1024_h8192", 1024, 8192),
    ("video_r2520_h8192", 2520, 8192),
    ("prefill_r64_h12288", 64, 12288),
    ("prefill_r128_h12288", 128, 12288),
    ("prefill_r256_h12288", 256, 12288),
    ("video_r1024_h12288", 1024, 12288),
    ("video_r2520_h12288", 2520, 12288),
]

NVFP4_LAYOUT_SHAPES = [
    ("rows1_d4096", 1, 4096),
    ("rows2_d4096", 2, 4096),
    ("rows31_d4096", 31, 4096),
    ("rows32_d4096", 32, 4096),
    ("rows33_d4096", 33, 4096),
    ("rows127_d4096", 127, 4096),
    ("rows128_d4096", 128, 4096),
    ("rows129_d4096", 129, 4096),
    ("rows16_d1024", 16, 1024),
    ("rows16_d2048", 16, 2048),
    ("rows16_d8192", 16, 8192),
    ("rows16_d12288", 16, 12288),
    ("rows64_d16384", 64, 16384),
]

SMALLM_DECODE_SHAPES = [
    ("k4096_n1024", 4096, 1024),
    ("k4096_n4096", 4096, 4096),
    ("k4096_n12288", 4096, 12288),
    ("k12288_n1024", 12288, 1024),
    ("k12288_n4096", 12288, 4096),
    ("k12288_n12288", 12288, 12288),
]

VLA_QK_ROWS = [1, 4, 8, 16, 24, 32, 48, 64, 128, 256]
VLA_QKV_TOKENS = [1, 4, 16, 64, 256, 1024, 2520, 4096]
VLA_QKV_HEADS = [8, 16, 24, 32, 48]

QUICK = {
    "fp8": FP8_QUANT_SHAPES[:3],
    "bf16_linear": BF16_LINEAR_SHAPES[:4],
    "fp8_ffn": [FP8_FFN_SHAPES[0], FP8_FFN_SHAPES[2]],
    "fused": FUSED_QUANT_SHAPES[:4],
    "layout": NVFP4_LAYOUT_SHAPES[:4],
    "smallm": [("k4096_n64", 4096, 64), ("k12288_n64", 12288, 64)],
    "vla_rows": [1, 24, 48],
    "vla_tokens": [1, 64, 256],
    "vla_heads": [24],
}

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)

SOURCE_SPECS = {
    "flashrt-gemm-epilogues": {
        "module": "flashrt_gemm_epilogues",
        "sources": [
            "torch-ext/torch_binding.cpp",
            "csrc/bf16_gemm_bias_gelu.cu",
            "csrc/bias_gelu_quantize_fp8.cu",
            "csrc/channel_scale_quantize_fp8.cu",
        ],
    },
    "flashrt-fp8-ffn": {
        "module": "flashrt_fp8_ffn",
        "sources": [
            "torch-ext/torch_binding.cpp",
            "csrc/fp8_ffn.cu",
        ],
    },
    "flashrt-vla-video": {
        "module": "flashrt_vla_video",
        "sources": [
            "torch-ext/torch_binding.cpp",
            "csrc/q_norm_rope_bf16.cu",
        ],
    },
    "flashrt-nvfp4": {
        "module": "flashrt_nvfp4",
        "sources": [
            "torch-ext/torch_binding.cpp",
            "csrc/nvfp4_sf_reshape_sm120.cu",
        ],
    },
    "flashrt-smallm-gemm": {
        "module": "flashrt_smallm_gemm",
        "sources": [
            "torch-ext/torch_binding.cpp",
            "csrc/fp4_w4a4_matvec_sm120.cu",
        ],
    },
    "flashrt-fused-quant": {
        "module": "flashrt_fused_quant",
        "sources": [
            "torch-ext/torch_binding.cpp",
            "csrc/silu_mul_to_nvfp4_swizzled.cu",
        ],
    },
}


@dataclass
class Result:
    package: str
    op: str
    shape: str
    status: str
    max_abs: float | None = None
    mean_abs: float | None = None
    p99_abs: float | None = None
    max_rel: float | None = None
    p99_rel: float | None = None
    cosine_similarity: float | None = None
    mismatches: int | None = None
    got_dtype: str | None = None
    expected_dtype: str | None = None
    tolerance: str | None = None
    note: str = ""


def _torch():
    import torch

    return torch


def _import(name: str):
    return importlib.import_module(name)


def _preload_cublaslt() -> None:
    torch = _torch()
    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    library = ctypes.util.find_library("cublasLt")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


class SourceOps:
    def __init__(self, namespace: str):
        self._ops = getattr(_torch().ops, namespace)

    def bf16_linear_bf16(self, x, w, out=None):
        torch = _torch()
        if out is None:
            out = torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=torch.bfloat16)
        self._ops.bf16_linear_bf16(x, w, out)
        return out

    def bf16_linear_bias_bf16(self, x, w, bias, out=None):
        torch = _torch()
        if out is None:
            out = torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=torch.bfloat16)
        self._ops.bf16_linear_bias_bf16(x, w, bias, out)
        return out

    def bf16_gemm_bias_gelu(self, a, b, bias, out=None):
        torch = _torch()
        if out is None:
            out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)
        self._ops.bf16_gemm_bias_gelu(a, b, bias, out)
        return out

    def bf16_gemm_bias(self, a, b, bias, out=None):
        torch = _torch()
        if out is None:
            out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)
        self._ops.bf16_gemm_bias(a, b, bias, out)
        return out

    def bias_gelu_quantize_fp8_static_bf16(self, input, bias, scale, out=None):
        torch = _torch()
        if out is None:
            out = torch.empty(input.shape, device=input.device, dtype=torch.float8_e4m3fn)
        self._ops.bias_gelu_quantize_fp8_static_bf16(input, bias, scale, out)
        return out

    def gelu_quantize_fp8_static_bf16(self, input, scale, out=None):
        torch = _torch()
        if out is None:
            out = torch.empty(input.shape, device=input.device, dtype=torch.float8_e4m3fn)
        self._ops.gelu_quantize_fp8_static_bf16(input, scale, out)
        return out

    def channel_scale_quantize_fp8_static_bf16(self, input, channel_scale, scale, out=None):
        torch = _torch()
        if out is None:
            out = torch.empty(input.shape, device=input.device, dtype=torch.float8_e4m3fn)
        self._ops.channel_scale_quantize_fp8_static_bf16(input, channel_scale, scale, out)
        return out

    def fp8_gemm_bf16(self, input, weight, input_scale, weight_scale, out=None):
        torch = _torch()
        if out is None:
            out = torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=torch.bfloat16)
        self._ops.fp8_gemm_bf16(input, weight, input_scale, weight_scale, out)
        return out

    def fp8_linear_bias_gelu_quant_bf16(
        self, input, weight, bias, input_scale, weight_scale, output_scale, hidden_bf16=None, out_fp8=None
    ):
        torch = _torch()
        if hidden_bf16 is None:
            hidden_bf16 = torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=torch.bfloat16)
        if out_fp8 is None:
            out_fp8 = torch.empty_like(hidden_bf16, dtype=torch.float8_e4m3fn)
        self._ops.fp8_linear_bias_gelu_quant_bf16(
            input, weight, bias, input_scale, weight_scale, output_scale, hidden_bf16, out_fp8
        )
        return hidden_bf16, out_fp8

    def fp8_gelu_mlp_bf16(
        self,
        input,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        input_scale,
        up_weight_scale,
        hidden_scale,
        down_weight_scale,
        *,
        hidden_bf16=None,
        hidden_fp8=None,
        out=None,
    ):
        torch = _torch()
        if hidden_bf16 is None:
            hidden_bf16 = torch.empty((input.shape[0], up_weight.shape[0]), device=input.device, dtype=torch.bfloat16)
        if hidden_fp8 is None:
            hidden_fp8 = torch.empty_like(hidden_bf16, dtype=torch.float8_e4m3fn)
        if out is None:
            out = torch.empty((input.shape[0], down_weight.shape[0]), device=input.device, dtype=torch.bfloat16)
        self._ops.fp8_gelu_mlp_bf16(
            input,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            input_scale,
            up_weight_scale,
            hidden_scale,
            down_weight_scale,
            hidden_bf16,
            hidden_fp8,
            out,
        )
        return out

    def q_norm_rope_bf16(self, q, weight, cos, sin, out=None, *, eps=1e-6):
        torch = _torch()
        if out is None:
            out = torch.empty_like(q)
        self._ops.q_norm_rope_bf16(q, weight, cos, sin, out, eps)
        return out

    def k_norm_rope_v_cache_bf16(self, k, v, weight, cos, sin, k_out=None, v_out=None, *, eps=1e-6):
        torch = _torch()
        if k_out is None:
            k_out = torch.empty_like(k)
        if v_out is None:
            v_out = torch.empty_like(v)
        self._ops.k_norm_rope_v_cache_bf16(k, v, weight, cos, sin, k_out, v_out, eps)
        return k_out, v_out

    def qkv_split_norm_rope_bf16(
        self, packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im, *, heads, head_dim, seq_len=None, q_out=None, k_out=None, eps=1e-6
    ):
        torch = _torch()
        if seq_len is None:
            seq_len = packed_qkv.shape[1]
        if q_out is None:
            q_out = torch.empty((packed_qkv.shape[0], packed_qkv.shape[1], heads, head_dim), device=packed_qkv.device, dtype=packed_qkv.dtype)
        if k_out is None:
            k_out = torch.empty_like(q_out)
        self._ops.qkv_split_norm_rope_bf16(
            packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im, q_out, k_out, heads, head_dim, seq_len, eps
        )
        return q_out, k_out

    def nvfp4_sf_linear_to_swizzled(self, scales, *, out=None, is_sfb=False):
        torch = _torch()
        rows = scales.shape[0]
        D = scales.shape[1] * 16
        if out is None:
            out = torch.zeros((_swizzled_bytes(rows, D),), device=scales.device, dtype=torch.uint8)
        self._ops.nvfp4_sf_linear_to_swizzled(scales, out, D, is_sfb)
        return out

    def nvfp4_w4a4_decode_matvec_bf16out(self, a_packed, b_packed, sfa, sfb, *, alpha=1.0, out=None):
        torch = _torch()
        if out is None:
            out = torch.empty((b_packed.shape[0],), device=b_packed.device, dtype=torch.bfloat16)
        self._ops.nvfp4_w4a4_decode_matvec_bf16out(a_packed, b_packed, sfa, sfb, out, float(alpha))
        return out

    def silu_mul_quant_nvfp4_swizzled_bf16(self, gate, up, *, packed=None, scales=None):
        torch = _torch()
        rows, cols = gate.shape
        if packed is None:
            packed = torch.empty((rows, cols // 2), device=gate.device, dtype=torch.uint8)
        if scales is None:
            scales = torch.zeros((_swizzled_bytes(rows, cols),), device=gate.device, dtype=torch.uint8)
        self._ops.silu_mul_quant_nvfp4_swizzled_bf16(gate, up, packed, scales)
        return packed, scales

    def silu_mul_merged_quant_nvfp4_swizzled_bf16(self, merged_gate_up, *, packed=None, scales=None):
        torch = _torch()
        rows, merged_cols = merged_gate_up.shape
        cols = merged_cols // 2
        if packed is None:
            packed = torch.empty((rows, cols // 2), device=merged_gate_up.device, dtype=torch.uint8)
        if scales is None:
            scales = torch.zeros((_swizzled_bytes(rows, cols),), device=merged_gate_up.device, dtype=torch.uint8)
        self._ops.silu_mul_merged_quant_nvfp4_swizzled_bf16(merged_gate_up, packed, scales)
        return packed, scales


def _load_source_ops(package: str):
    from torch.utils.cpp_extension import load

    spec = SOURCE_SPECS[package]
    pkg_dir = ROOT / package
    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    namespace = f"flashrt_accuracy_{spec['module']}"
    if package in {"flashrt-nvfp4", "flashrt-smallm-gemm", "flashrt-fused-quant"}:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    load(
        name=namespace,
        sources=[str(pkg_dir / item) for item in spec["sources"]],
        extra_include_paths=[str(pkg_dir / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        verbose=False,
    )
    return SourceOps(namespace)


def _load_ops(package: str, module_name: str, backend: str):
    if backend == "installed":
        return _import(module_name)
    return _load_source_ops(package)


def _require_cuda(results: list[Result], package: str) -> bool:
    torch = _torch()
    if not torch.cuda.is_available():
        results.append(Result(package, "*", "*", "BLOCKED", note="CUDA is required"))
        return False
    return True


def _cosine_similarity(got_f, exp_f) -> float:
    torch = _torch()
    got_flat = got_f.flatten()
    exp_flat = exp_f.flatten()
    if got_flat.numel() == 0:
        return 1.0
    got_norm = torch.linalg.vector_norm(got_flat)
    exp_norm = torch.linalg.vector_norm(exp_flat)
    if got_norm.item() == 0.0 or exp_norm.item() == 0.0:
        return 1.0 if bool(torch.equal(got_flat, exp_flat)) else 0.0
    return float(torch.nn.functional.cosine_similarity(got_flat, exp_flat, dim=0).item())


def _result_exact(package: str, op: str, shape: str, got, expected) -> Result:
    torch = _torch()
    got_cpu = got.detach().cpu()
    expected_cpu = expected.detach().cpu()
    mismatches = int((got_cpu != expected_cpu).sum().item())
    got_f = got.detach().float()
    exp_f = expected.detach().float()
    abs_err = (got_f - exp_f).abs().flatten()
    max_abs = float(abs_err.max().item()) if abs_err.numel() else 0.0
    mean_abs = float(abs_err.mean().item()) if abs_err.numel() else 0.0
    return Result(
        package,
        op,
        shape,
        "PASS" if mismatches == 0 else "FAIL",
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=max_abs,
        max_rel=0.0 if mismatches == 0 else None,
        cosine_similarity=_cosine_similarity(got_f, exp_f),
        mismatches=mismatches,
        got_dtype=str(got.dtype),
        expected_dtype=str(expected.dtype),
        tolerance="exact byte/value parity",
        note="byte parity" if mismatches == 0 else "exact output mismatch",
    )


def _result_approx(
    package: str,
    op: str,
    shape: str,
    got,
    expected,
    *,
    max_abs_limit: float,
    max_rel_limit: float,
    rel_floor: float,
    max_ulp_limit: int | None = None,
) -> Result:
    torch = _torch()
    got_f = got.detach().float()
    exp_f = expected.detach().float()
    abs_err = (got_f - exp_f).abs().flatten()
    rel_err = abs_err / exp_f.abs().clamp_min(rel_floor).flatten()
    max_abs = float(abs_err.max().item()) if abs_err.numel() else 0.0
    mean_abs = float(abs_err.mean().item()) if abs_err.numel() else 0.0
    if abs_err.numel():
        max_idx = int(abs_err.argmax().item())
        got_at_max = float(got_f.flatten()[max_idx].item())
        exp_at_max = float(exp_f.flatten()[max_idx].item())
    else:
        max_idx = -1
        got_at_max = 0.0
        exp_at_max = 0.0
    if abs_err.numel():
        kth = max(1, math.ceil(0.99 * abs_err.numel()))
        p99_abs = float(abs_err.kthvalue(kth).values.item())
    else:
        p99_abs = 0.0
    max_rel = float(rel_err.max().item()) if rel_err.numel() else 0.0
    max_ulp = None
    if got.dtype == torch.bfloat16 and expected.dtype == torch.bfloat16:
        got_bits = got.detach().cpu().view(torch.int16).to(torch.int32) & 0xFFFF
        exp_bits = expected.detach().cpu().view(torch.int16).to(torch.int32) & 0xFFFF
        got_ordered = torch.where((got_bits & 0x8000) != 0, 0x8000 - (got_bits & 0x7FFF), got_bits)
        exp_ordered = torch.where((exp_bits & 0x8000) != 0, 0x8000 - (exp_bits & 0x7FFF), exp_bits)
        max_ulp = int((got_ordered - exp_ordered).abs().max().item())
    if max_ulp_limit is not None and max_ulp is not None:
        passed = max_ulp <= max_ulp_limit
    else:
        passed = max_abs <= max_abs_limit and max_rel <= max_rel_limit
    limit_note = f"limits max_abs<={max_abs_limit:g}, max_rel<={max_rel_limit:g}, rel_floor={rel_floor:g}"
    if max_ulp_limit is not None:
        limit_note += f", max_ulp<={max_ulp_limit}"
    return Result(
        package,
        op,
        shape,
        "PASS" if passed else "FAIL",
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        max_rel=max_rel,
        cosine_similarity=_cosine_similarity(got_f, exp_f),
        got_dtype=str(got.dtype),
        expected_dtype=str(expected.dtype),
        tolerance=limit_note,
        note=(
            f"{limit_note}, max_ulp={max_ulp}, max_idx={max_idx}, "
            f"got={got_at_max:.8g}, expected={exp_at_max:.8g}"
        ),
    )


def _result_allclose(
    package: str,
    op: str,
    shape: str,
    got,
    expected,
    *,
    atol: float,
    rtol: float,
    rel_floor: float,
) -> Result:
    torch = _torch()
    got_f = got.detach().float()
    exp_f = expected.detach().float()
    abs_err = (got_f - exp_f).abs().flatten()
    rel_err = abs_err / exp_f.abs().clamp_min(rel_floor).flatten()
    max_abs = float(abs_err.max().item()) if abs_err.numel() else 0.0
    mean_abs = float(abs_err.mean().item()) if abs_err.numel() else 0.0
    max_rel = float(rel_err.max().item()) if rel_err.numel() else 0.0
    if abs_err.numel():
        kth = max(1, math.ceil(0.99 * abs_err.numel()))
        p99_abs = float(abs_err.kthvalue(kth).values.item())
        p99_rel = float(rel_err.kthvalue(kth).values.item())
    else:
        p99_abs = 0.0
        p99_rel = 0.0
    passed = bool(torch.allclose(got_f, exp_f, atol=atol, rtol=rtol))
    tolerance = f"torch.allclose(atol={atol:g}, rtol={rtol:g}), rel_floor={rel_floor:g}"
    return Result(
        package,
        op,
        shape,
        "PASS" if passed else "FAIL",
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        max_rel=max_rel,
        p99_rel=p99_rel,
        cosine_similarity=_cosine_similarity(got_f, exp_f),
        got_dtype=str(got.dtype),
        expected_dtype=str(expected.dtype),
        tolerance=tolerance,
        note=tolerance,
    )


def _result_p99_approx(
    package: str,
    op: str,
    shape: str,
    got,
    expected,
    *,
    p99_abs_limit: float,
    p99_rel_limit: float,
    rel_floor: float,
) -> Result:
    got_f = got.detach().float()
    exp_f = expected.detach().float()
    abs_err = (got_f - exp_f).abs().flatten()
    rel_err = abs_err / exp_f.abs().clamp_min(rel_floor).flatten()
    max_abs = float(abs_err.max().item()) if abs_err.numel() else 0.0
    mean_abs = float(abs_err.mean().item()) if abs_err.numel() else 0.0
    max_rel = float(rel_err.max().item()) if rel_err.numel() else 0.0
    if abs_err.numel():
        kth = max(1, math.ceil(0.99 * abs_err.numel()))
        p99_abs = float(abs_err.kthvalue(kth).values.item())
        p99_rel = float(rel_err.kthvalue(kth).values.item())
    else:
        p99_abs = 0.0
        p99_rel = 0.0
    passed = p99_abs <= p99_abs_limit and p99_rel <= p99_rel_limit
    tolerance = (
        f"p99_abs<={p99_abs_limit:g}, p99_rel_floor{rel_floor:g}<={p99_rel_limit:g}"
    )
    return Result(
        package,
        op,
        shape,
        "PASS" if passed else "FAIL",
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        max_rel=max_rel,
        p99_rel=p99_rel,
        cosine_similarity=_cosine_similarity(got_f, exp_f),
        got_dtype=str(got.dtype),
        expected_dtype=str(expected.dtype),
        tolerance=tolerance,
        note=tolerance,
    )


def _result_p99_cosine(
    package: str,
    op: str,
    shape: str,
    got,
    expected,
    *,
    p99_abs_limit: float,
    cosine_limit: float,
) -> Result:
    got_f = got.detach().float()
    exp_f = expected.detach().float()
    abs_err = (got_f - exp_f).abs().flatten()
    max_abs = float(abs_err.max().item()) if abs_err.numel() else 0.0
    mean_abs = float(abs_err.mean().item()) if abs_err.numel() else 0.0
    if abs_err.numel():
        kth = max(1, math.ceil(0.99 * abs_err.numel()))
        p99_abs = float(abs_err.kthvalue(kth).values.item())
    else:
        p99_abs = 0.0
    cosine = _cosine_similarity(got_f, exp_f)
    passed = p99_abs <= p99_abs_limit and cosine >= cosine_limit
    tolerance = f"p99_abs<={p99_abs_limit:g}, cosine>={cosine_limit:g}"
    return Result(
        package,
        op,
        shape,
        "PASS" if passed else "FAIL",
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine_similarity=cosine,
        got_dtype=str(got.dtype),
        expected_dtype=str(expected.dtype),
        tolerance=tolerance,
        note=tolerance,
    )


def _result_fp8_quant_distribution(
    package: str,
    op: str,
    shape: str,
    got,
    expected,
    *,
    p99_abs_limit: float,
    mismatch_rate_limit: float,
) -> Result:
    got_f = got.detach().float()
    exp_f = expected.detach().float()
    abs_err = (got_f - exp_f).abs().flatten()
    max_abs = float(abs_err.max().item()) if abs_err.numel() else 0.0
    mean_abs = float(abs_err.mean().item()) if abs_err.numel() else 0.0
    if abs_err.numel():
        kth = max(1, math.ceil(0.99 * abs_err.numel()))
        p99_abs = float(abs_err.kthvalue(kth).values.item())
    else:
        p99_abs = 0.0
    mismatches = int((got.detach().cpu() != expected.detach().cpu()).sum().item())
    mismatch_rate = mismatches / max(1, got.numel())
    passed = p99_abs <= p99_abs_limit and mismatch_rate <= mismatch_rate_limit
    tolerance = f"p99_abs<={p99_abs_limit:g}, mismatch_rate<={mismatch_rate_limit:g}"
    return Result(
        package,
        op,
        shape,
        "PASS" if passed else "FAIL",
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine_similarity=_cosine_similarity(got_f, exp_f),
        mismatches=mismatches,
        got_dtype=str(got.dtype),
        expected_dtype=str(expected.dtype),
        tolerance=tolerance,
        note=f"{tolerance}, mismatch_rate={mismatch_rate:.8g}",
    )


def _swizzled_bytes(rows: int, cols: int) -> int:
    n_blocks = cols // 16
    return ((rows + 127) // 128) * ((n_blocks + 3) // 4) * 512


def _reference_swizzle(scales):
    torch = _torch()
    rows, n_blocks = scales.shape
    n_col_super = (n_blocks + 3) // 4
    out = torch.zeros(_swizzled_bytes(rows, n_blocks * 16), dtype=torch.uint8)
    src = scales.detach().cpu()
    for row in range(rows):
        rb = row // 128
        ri = row % 128
        for block in range(n_blocks):
            cb = block // 4
            ci = block % 4
            super_idx = rb * n_col_super + cb
            inner_off = (ri % 32) * 16 + (ri // 32) * 4 + ci
            out[super_idx * 512 + inner_off] = src[row, block]
    return out


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


def _scale_tables():
    torch = _torch()
    values: list[float] = []
    bytes_: list[int] = []
    for b in range(1, 0x78):
        values.append(_ue4m3_to_float(b))
        bytes_.append(b)
    values.append(_ue4m3_to_float(0xFE))
    bytes_.append(0xFE)
    return torch.tensor(values, dtype=torch.float32), torch.tensor(bytes_, dtype=torch.uint8)


def _float_to_fp4_e2m1_tensor(v):
    torch = _torch()
    a = v.abs()
    mag = torch.zeros_like(a, dtype=torch.uint8)
    mag = torch.where(a >= 0.25, torch.tensor(1, dtype=torch.uint8), mag)
    mag = torch.where(a >= 0.75, torch.tensor(2, dtype=torch.uint8), mag)
    mag = torch.where(a >= 1.25, torch.tensor(3, dtype=torch.uint8), mag)
    mag = torch.where(a >= 1.75, torch.tensor(4, dtype=torch.uint8), mag)
    mag = torch.where(a >= 2.5, torch.tensor(5, dtype=torch.uint8), mag)
    mag = torch.where(a >= 3.5, torch.tensor(6, dtype=torch.uint8), mag)
    mag = torch.where(a >= 5.0, torch.tensor(7, dtype=torch.uint8), mag)
    sign = torch.where(v < 0, torch.tensor(8, dtype=torch.uint8), torch.tensor(0, dtype=torch.uint8))
    return sign | mag


def _reference_silu_nvfp4(gate, up):
    torch = _torch()
    rows, cols = gate.shape
    values = torch.nn.functional.silu(gate.float()).to(torch.bfloat16)
    values = (values.float() * up.float()).to(torch.bfloat16).float().cpu()
    blocks = values.reshape(rows, cols // 16, 16)
    amax = blocks.abs().amax(dim=2)
    targets = amax / 6.0
    table_values, table_bytes = _scale_tables()
    flat = targets.flatten()
    idx = torch.bucketize(flat, table_values, right=False).clamp_max(table_values.numel() - 1)
    scale_linear = table_bytes[idx].reshape_as(targets)
    scale_linear = torch.where(targets <= 0, torch.zeros_like(scale_linear), scale_linear)
    decoded = torch.tensor([_ue4m3_to_float(int(b)) for b in scale_linear.flatten()], dtype=torch.float32)
    decoded = decoded.reshape_as(targets)
    inv = torch.where(decoded > 0, 1.0 / decoded, torch.zeros_like(decoded))
    quant = blocks * inv[..., None]
    lo = _float_to_fp4_e2m1_tensor(quant[..., 0::2])
    hi = _float_to_fp4_e2m1_tensor(quant[..., 1::2])
    packed = ((hi << 4) | lo).reshape(rows, cols // 2).contiguous()
    return packed, _reference_swizzle(scale_linear)


def _fp4_codebook_tensor():
    torch = _torch()
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )


def _ue4m3_lut_tensor():
    torch = _torch()
    return torch.tensor([_ue4m3_to_float(i) for i in range(256)], dtype=torch.float32)


def _unpack_fp4(packed):
    torch = _torch()
    codebook = _fp4_codebook_tensor().to(packed.device)
    lo = packed & 0x0F
    hi = packed >> 4
    out = torch.empty((packed.shape[0], packed.shape[1] * 2), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = codebook[lo.long()]
    out[:, 1::2] = codebook[hi.long()]
    return out


def _deswizzle_scales(swizzled, rows: int, cols: int):
    torch = _torch()
    n_blocks = cols // 16
    n_col_super = (n_blocks + 3) // 4
    out = torch.empty((rows, n_blocks), device=swizzled.device, dtype=torch.uint8)
    for row in range(rows):
        rb = row // 128
        ri = row % 128
        for block in range(n_blocks):
            cb = block // 4
            ci = block % 4
            super_idx = rb * n_col_super + cb
            inner_off = (ri % 32) * 16 + (ri // 32) * 4 + ci
            out[row, block] = swizzled[super_idx * 512 + inner_off]
    return out


def _reference_smallm(a_packed, b_packed, sfa_linear, sfb_linear, K: int, alpha: float, chunk_rows: int):
    torch = _torch()
    device = b_packed.device
    N = b_packed.shape[0]
    lut = _ue4m3_lut_tensor().to(device)
    a = _unpack_fp4(a_packed.reshape(1, -1)).reshape(K)
    a_scale = lut[sfa_linear.reshape(-1).to(device).long()].repeat_interleave(16)
    a = a * a_scale
    sfb_linear = sfb_linear.to(device)
    out = torch.empty((N,), device=device, dtype=torch.bfloat16)
    for start in range(0, N, chunk_rows):
        end = min(start + chunk_rows, N)
        b = _unpack_fp4(b_packed[start:end])
        b_scale = lut[sfb_linear[start:end].long()].repeat_interleave(16, dim=1)
        expected = (b * b_scale * a.reshape(1, K)).sum(dim=1) * alpha
        out[start:end] = expected.to(torch.bfloat16)
    return out


def _quantize_fp8(x, scale):
    torch = _torch()
    return torch.clamp(x.float() / scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)


def _dequant_fp8(x, scale):
    return x.float() * scale.float()


def _reference_fp8_gemm(x, w, x_scale, w_scale):
    torch = _torch()
    return (_dequant_fp8(x, x_scale) @ _dequant_fp8(w, w_scale).T).to(torch.bfloat16)


def _reference_fp8_linear_bias_gelu_quant(x, w, bias, x_scale, w_scale, y_scale):
    torch = _torch()
    hidden = _reference_fp8_gemm(x, w, x_scale, w_scale)
    y = torch.nn.functional.gelu(hidden.float() + bias.float(), approximate="tanh")
    y_fp8 = torch.clamp(y / y_scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)
    return hidden, y_fp8


def _reference_fp8_mlp(x, up_w, up_b, down_w, down_b, x_scale, up_w_scale, hidden_scale, down_w_scale):
    torch = _torch()
    _, hidden_fp8 = _reference_fp8_linear_bias_gelu_quant(
        x, up_w, up_b, x_scale, up_w_scale, hidden_scale
    )
    out = _reference_fp8_gemm(hidden_fp8, down_w, hidden_scale, down_w_scale)
    return (out.float() + down_b.float()).to(torch.bfloat16)


def _make_fp8_ffn_case(m: int, k: int, h: int, n: int):
    torch = _torch()
    x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    up_w_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    hidden_scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    down_w_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    x = _quantize_fp8(torch.randn((m, k), device="cuda", dtype=torch.bfloat16), x_scale)
    up_w = _quantize_fp8(torch.randn((h, k), device="cuda", dtype=torch.bfloat16), up_w_scale)
    down_w = _quantize_fp8(torch.randn((n, h), device="cuda", dtype=torch.bfloat16), down_w_scale)
    up_b = torch.randn((h,), device="cuda", dtype=torch.bfloat16)
    down_b = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    return x, up_w, up_b, down_w, down_b, x_scale, up_w_scale, hidden_scale, down_w_scale


def _reference_norm_rope(x, weight, cos, sin, eps=1e-6):
    torch = _torch()
    half = x.shape[-1] // 2
    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    normed = x.float() * rstd * weight.float()
    lo = normed[..., :half]
    hi = normed[..., half:]
    out_lo = lo * cos.float() - hi * sin.float()
    out_hi = hi * cos.float() + lo * sin.float()
    return torch.cat([out_lo, out_hi], dim=-1).to(torch.bfloat16)


def _reference_qkv_split_norm_rope(
    packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im, heads, head_dim, eps=1e-6
):
    torch = _torch()
    batch, tokens, _ = packed_qkv.shape
    dim = heads * head_dim
    q = packed_qkv[..., :dim].reshape(batch, tokens, heads, head_dim)
    k = packed_qkv[..., dim : 2 * dim].reshape(batch, tokens, heads, head_dim)
    qf = q.float()
    kf = k.float()
    qn = qf * torch.rsqrt((qf * qf).mean(dim=(-2, -1), keepdim=True) + eps)
    kn = kf * torch.rsqrt((kf * kf).mean(dim=(-2, -1), keepdim=True) + eps)
    qn = qn * norm_q_weight.reshape(1, 1, heads, head_dim).float()
    kn = kn * norm_k_weight.reshape(1, 1, heads, head_dim).float()

    def rope(x):
        xr = x[..., 0::2]
        xi = x[..., 1::2]
        fr = freqs_re[:tokens][None, :, None, :]
        fi = freqs_im[:tokens][None, :, None, :]
        out = torch.empty_like(x, dtype=torch.float32)
        out[..., 0::2] = xr * fr - xi * fi
        out[..., 1::2] = xr * fi + xi * fr
        return out.to(torch.bfloat16)

    return rope(qn), rope(kn)


def sweep_gemm_epilogues(args, results: list[Result]) -> None:
    package = "flashrt-gemm-epilogues"
    if not _require_cuda(results, package):
        return
    torch = _torch()
    if not hasattr(torch, "float8_e4m3fn"):
        results.append(Result(package, "*", "*", "BLOCKED", note="torch.float8_e4m3fn is required"))
        return
    try:
        ops = _load_ops(package, "flashrt_gemm_epilogues", args.backend)
    except Exception as exc:
        results.append(Result(package, "*", "*", "BLOCKED", note=f"import failed: {exc}"))
        return
    shapes = QUICK["fp8"] if args.mode == "quick" else FP8_QUANT_SHAPES
    for label, m, n in shapes:
        torch.manual_seed(100 + m + n)
        x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16).contiguous()
        bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16).contiguous()
        scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)
        expected = torch.clamp(
            torch.nn.functional.gelu(x.float() + bias.float(), approximate="tanh") / scale,
            -448.0,
            448.0,
        ).to(torch.float8_e4m3fn)
        got = ops.bias_gelu_quantize_fp8_static_bf16(x, bias, scale)
        results.append(_result_exact(package, "bias_gelu_quantize_fp8_static_bf16", label, got, expected))

        expected = torch.clamp(torch.nn.functional.gelu(x.float(), approximate="tanh") / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
        got = ops.gelu_quantize_fp8_static_bf16(x, scale)
        results.append(_result_exact(package, "gelu_quantize_fp8_static_bf16", label, got, expected))

        channel_scale = torch.randn((n,), device="cuda", dtype=torch.bfloat16).contiguous()
        expected = torch.clamp((x.float() * channel_scale.float()) / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
        got = ops.channel_scale_quantize_fp8_static_bf16(x, channel_scale, scale)
        results.append(_result_exact(package, "channel_scale_quantize_fp8_static_bf16", label, got, expected))

    linear_shapes = QUICK["bf16_linear"] if args.mode == "quick" else BF16_LINEAR_SHAPES
    for label, m, k, n in linear_shapes:
        torch.manual_seed(11000 + m + k + n)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).contiguous()
        w = torch.randn((k, n), device="cuda", dtype=torch.bfloat16).contiguous()
        bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16).contiguous()
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

        expected = (x @ w).to(torch.bfloat16)
        got = ops.bf16_linear_bf16(x, w, out=out)
        results.append(_result_p99_cosine(
            package,
            "bf16_linear_bf16",
            label,
            got,
            expected,
            p99_abs_limit=0.5,
            cosine_limit=0.999,
        ))

        expected = torch.addmm(bias, x, w).to(torch.bfloat16)
        got = ops.bf16_linear_bias_bf16(x, w, bias, out=out)
        results.append(_result_p99_cosine(
            package,
            "bf16_linear_bias_bf16",
            label,
            got,
            expected,
            p99_abs_limit=0.5,
            cosine_limit=0.999,
        ))


def sweep_fp8_ffn(args, results: list[Result]) -> None:
    package = "flashrt-fp8-ffn"
    if not _require_cuda(results, package):
        return
    torch = _torch()
    if not hasattr(torch, "float8_e4m3fn"):
        results.append(Result(package, "*", "*", "BLOCKED", note="torch.float8_e4m3fn is required"))
        return
    try:
        ops = _load_ops(package, "flashrt_fp8_ffn", args.backend)
    except Exception as exc:
        results.append(Result(package, "*", "*", "BLOCKED", note=f"import failed: {exc}"))
        return
    shapes = QUICK["fp8_ffn"] if args.mode == "quick" else FP8_FFN_SHAPES
    for label, m, k, h, n in shapes:
        torch.manual_seed(900 + m + k + h + n)
        x, up_w, up_b, down_w, down_b, x_s, up_s, hid_s, dn_s = _make_fp8_ffn_case(m, k, h, n)

        got_gemm = ops.fp8_gemm_bf16(x, up_w, x_s, up_s)
        exp_gemm = _reference_fp8_gemm(x, up_w, x_s, up_s)
        results.append(_result_allclose(package, "fp8_gemm_bf16", label, got_gemm, exp_gemm, atol=0.25, rtol=0.03, rel_floor=args.rel_floor))

        _, got_hidden_fp8 = ops.fp8_linear_bias_gelu_quant_bf16(x, up_w, up_b, x_s, up_s, hid_s)
        _, exp_hidden_fp8 = _reference_fp8_linear_bias_gelu_quant(x, up_w, up_b, x_s, up_s, hid_s)
        results.append(_result_fp8_quant_distribution(package, "fp8_linear_bias_gelu_quant_bf16", label, got_hidden_fp8, exp_hidden_fp8, p99_abs_limit=0.0, mismatch_rate_limit=1e-4))

        got_mlp = ops.fp8_gelu_mlp_bf16(x, up_w, up_b, down_w, down_b, x_s, up_s, hid_s, dn_s)
        exp_mlp = _reference_fp8_mlp(x, up_w, up_b, down_w, down_b, x_s, up_s, hid_s, dn_s)
        results.append(_result_p99_approx(package, "fp8_gelu_mlp_bf16", label, got_mlp, exp_mlp, p99_abs_limit=1.0, p99_rel_limit=0.05, rel_floor=args.rel_floor))


def sweep_vla(args, results: list[Result]) -> None:
    package = "flashrt-vla-video"
    if not _require_cuda(results, package):
        return
    torch = _torch()
    try:
        ops = _load_ops(package, "flashrt_vla_video", args.backend)
    except Exception as exc:
        results.append(Result(package, "*", "*", "BLOCKED", note=f"import failed: {exc}"))
        return
    rows_grid = QUICK["vla_rows"] if args.mode == "quick" else VLA_QK_ROWS
    for rows in rows_grid:
        torch.manual_seed(200 + rows)
        x = (torch.randn((rows, 128), device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        v = (torch.randn_like(x) * 0.2).contiguous()
        weight = (torch.randn((128,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
        t = torch.arange(64, device="cuda", dtype=torch.float32) / 64.0
        cos = torch.cos(t).to(torch.bfloat16).contiguous()
        sin = torch.sin(t).to(torch.bfloat16).contiguous()
        expected = _reference_norm_rope(x, weight, cos, sin)
        got = ops.q_norm_rope_bf16(x, weight, cos, sin)
        results.append(_result_approx(package, "q_norm_rope_bf16", f"rows{rows}", got, expected, max_abs_limit=args.vla_max_abs, max_rel_limit=args.vla_max_rel, rel_floor=args.rel_floor))
        k_out, v_out = ops.k_norm_rope_v_cache_bf16(x, v, weight, cos, sin)
        results.append(_result_approx(package, "k_norm_rope_v_cache_bf16:k", f"rows{rows}", k_out, expected, max_abs_limit=args.vla_max_abs, max_rel_limit=args.vla_max_rel, rel_floor=args.rel_floor))
        results.append(_result_exact(package, "k_norm_rope_v_cache_bf16:v", f"rows{rows}", v_out, v))

    tokens_grid = QUICK["vla_tokens"] if args.mode == "quick" else VLA_QKV_TOKENS
    heads_grid = QUICK["vla_heads"] if args.mode == "quick" else VLA_QKV_HEADS
    for heads in heads_grid:
        for tokens in tokens_grid:
            torch.manual_seed(300 + tokens + heads)
            head_dim = 128
            dim = heads * head_dim
            packed = (torch.randn((1, tokens, 3 * dim), device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            q_weight = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
            k_weight = (torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
            pos = torch.arange(max(tokens, 1), device="cuda", dtype=torch.float32)[:, None]
            freq = torch.arange(head_dim // 2, device="cuda", dtype=torch.float32)[None, :]
            angles = pos / (10000.0 ** (2.0 * freq / head_dim))
            freqs_re = torch.cos(angles).contiguous()
            freqs_im = torch.sin(angles).contiguous()
            q_got, k_got = ops.qkv_split_norm_rope_bf16(
                packed, q_weight, k_weight, freqs_re, freqs_im, heads=heads, head_dim=head_dim, seq_len=tokens
            )
            q_exp, k_exp = _reference_qkv_split_norm_rope(packed, q_weight, k_weight, freqs_re, freqs_im, heads, head_dim)
            shape = f"b1_t{tokens}_h{heads}_d{head_dim}"
            results.append(_result_approx(package, "qkv_split_norm_rope_bf16:q", shape, q_got, q_exp, max_abs_limit=args.vla_max_abs, max_rel_limit=args.vla_max_rel, rel_floor=args.rel_floor))
            results.append(_result_approx(package, "qkv_split_norm_rope_bf16:k", shape, k_got, k_exp, max_abs_limit=args.vla_max_abs, max_rel_limit=args.vla_max_rel, rel_floor=args.rel_floor))


def sweep_nvfp4(args, results: list[Result]) -> None:
    package = "flashrt-nvfp4"
    if not _require_cuda(results, package):
        return
    torch = _torch()
    try:
        ops = _load_ops(package, "flashrt_nvfp4", args.backend)
    except Exception as exc:
        results.append(Result(package, "*", "*", "BLOCKED", note=f"import failed: {exc}"))
        return
    shapes = QUICK["layout"] if args.mode == "quick" else NVFP4_LAYOUT_SHAPES
    for label, rows, D in shapes:
        torch.manual_seed(400 + rows + D)
        scales_cpu = torch.randint(0, 256, (rows, D // 16), dtype=torch.uint8)
        got = ops.nvfp4_sf_linear_to_swizzled(scales_cpu.cuda())
        expected = _reference_swizzle(scales_cpu).cuda()
        results.append(_result_exact(package, "nvfp4_sf_linear_to_swizzled", label, got, expected))


def sweep_fused_quant(args, results: list[Result]) -> None:
    package = "flashrt-fused-quant"
    if not _require_cuda(results, package):
        return
    torch = _torch()
    try:
        ops = _load_ops(package, "flashrt_fused_quant", args.backend)
    except Exception as exc:
        results.append(Result(package, "*", "*", "BLOCKED", note=f"import failed: {exc}"))
        return
    shapes = QUICK["fused"] if args.mode == "quick" else FUSED_QUANT_SHAPES
    for label, rows, cols in shapes:
        torch.manual_seed(500 + rows + cols)
        gate = (torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        up = (torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()
        expected_packed, expected_scales = _reference_silu_nvfp4(gate, up)
        got_packed, got_scales = ops.silu_mul_quant_nvfp4_swizzled_bf16(gate, up)
        results.append(_result_exact(package, "silu_mul_quant_nvfp4_swizzled_bf16:packed", label, got_packed, expected_packed.cuda()))
        results.append(_result_exact(package, "silu_mul_quant_nvfp4_swizzled_bf16:scales", label, got_scales, expected_scales.cuda()))
        merged = torch.cat([gate, up], dim=1).contiguous()
        got_packed, got_scales = ops.silu_mul_merged_quant_nvfp4_swizzled_bf16(merged)
        results.append(_result_exact(package, "silu_mul_merged_quant_nvfp4_swizzled_bf16:packed", label, got_packed, expected_packed.cuda()))
        results.append(_result_exact(package, "silu_mul_merged_quant_nvfp4_swizzled_bf16:scales", label, got_scales, expected_scales.cuda()))


def sweep_smallm(args, results: list[Result]) -> None:
    package = "flashrt-smallm-gemm"
    if not _require_cuda(results, package):
        return
    torch = _torch()
    try:
        ops = _load_ops(package, "flashrt_smallm_gemm", args.backend)
    except Exception as exc:
        results.append(Result(package, "*", "*", "BLOCKED", note=f"import failed: {exc}"))
        return
    shapes = QUICK["smallm"] if args.mode == "quick" else SMALLM_DECODE_SHAPES
    for label, K, N in shapes:
        alpha = 0.5
        a_packed = torch.full((K // 2,), 0x11, device="cuda", dtype=torch.uint8)
        b_packed = torch.full((N, K // 2), 0x11, device="cuda", dtype=torch.uint8)
        sfa = _reference_swizzle(torch.full((1, K // 16), 0x38, dtype=torch.uint8)).cuda()
        sfb = _reference_swizzle(torch.full((N, K // 16), 0x38, dtype=torch.uint8)).cuda()
        got = ops.nvfp4_w4a4_decode_matvec_bf16out(a_packed, b_packed, sfa, sfb, alpha=alpha)
        expected = torch.full((N,), K * 0.25 * alpha, device="cuda", dtype=torch.bfloat16)
        results.append(_result_approx(package, "nvfp4_w4a4_decode_matvec_bf16out:constant", label, got, expected, max_abs_limit=args.smallm_max_abs, max_rel_limit=args.smallm_max_rel, rel_floor=args.rel_floor, max_ulp_limit=args.smallm_max_ulp))

        torch.manual_seed(600 + K + N)
        a_packed = torch.randint(0, 256, (K // 2,), device="cuda", dtype=torch.uint8)
        b_packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
        sfa_linear = torch.randint(0, 0x78, (1, K // 16), dtype=torch.uint8)
        sfb_linear = torch.randint(0, 0x78, (N, K // 16), dtype=torch.uint8)
        sfa = _reference_swizzle(sfa_linear).cuda()
        sfb = _reference_swizzle(sfb_linear).cuda()
        got = ops.nvfp4_w4a4_decode_matvec_bf16out(a_packed, b_packed, sfa, sfb, alpha=alpha)
        expected = _reference_smallm(a_packed, b_packed, sfa_linear, sfb_linear, K, alpha, args.smallm_chunk_rows)
        results.append(_result_approx(package, "nvfp4_w4a4_decode_matvec_bf16out:random", label, got, expected, max_abs_limit=args.smallm_max_abs, max_rel_limit=args.smallm_max_rel, rel_floor=args.rel_floor, max_ulp_limit=args.smallm_max_ulp))


def _selected_packages(value: str) -> list[str]:
    if value == "all":
        return [
            "flashrt-gemm-epilogues",
            "flashrt-fp8-ffn",
            "flashrt-vla-video",
            "flashrt-nvfp4",
            "flashrt-smallm-gemm",
            "flashrt-fused-quant",
        ]
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_results(results: Iterable[Result], *, quiet: bool) -> None:
    for r in results:
        if quiet and r.status == "PASS":
            continue
        metrics = []
        if r.mismatches is not None:
            metrics.append(f"mismatches={r.mismatches}")
        if r.max_abs is not None:
            metrics.append(f"max_abs={r.max_abs:.6g}")
        if r.mean_abs is not None:
            metrics.append(f"mean_abs={r.mean_abs:.6g}")
        if r.p99_abs is not None:
            metrics.append(f"p99_abs={r.p99_abs:.6g}")
        if r.max_rel is not None:
            metrics.append(f"max_rel={r.max_rel:.6g}")
        if r.p99_rel is not None:
            metrics.append(f"p99_rel={r.p99_rel:.6g}")
        if r.cosine_similarity is not None:
            metrics.append(f"cos={r.cosine_similarity:.9g}")
        if r.got_dtype is not None or r.expected_dtype is not None:
            metrics.append(f"dtype={r.got_dtype}->{r.expected_dtype}")
        if r.tolerance:
            metrics.append(f"tol={r.tolerance}")
        if r.note:
            metrics.append(r.note)
        print(f"{r.status:7s} {r.package:24s} {r.op:48s} {r.shape:24s} {'; '.join(metrics)}")


def _write_json(path: Path, results: list[Result], args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": args.backend,
        "mode": args.mode,
        "package": args.package,
        "result_count": len(results),
        "pass_count": sum(1 for r in results if r.status == "PASS"),
        "fail_count": sum(1 for r in results if r.status != "PASS"),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_markdown(path: Path, results: list[Result], args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FlashRT Kernel Correctness Report",
        "",
        f"- Backend: `{args.backend}`",
        f"- Mode: `{args.mode}`",
        f"- Package selection: `{args.package}`",
        f"- Checks: `{len(results)}`",
        f"- Passing: `{sum(1 for r in results if r.status == 'PASS')}`",
        f"- Non-passing: `{sum(1 for r in results if r.status != 'PASS')}`",
        "",
        "| Status | Package | Op | Shape | Got dtype | Expected dtype | Max abs | Mean abs | P99 abs | Max rel | P99 rel | Cosine | Tolerance | Note |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in results:
        def fmt(value: float | None) -> str:
            return "" if value is None else f"{value:.6g}"

        lines.append(
            "| "
            + " | ".join(
                [
                    r.status,
                    r.package,
                    r.op,
                    r.shape,
                    r.got_dtype or "",
                    r.expected_dtype or "",
                    fmt(r.max_abs),
                    fmt(r.mean_abs),
                    fmt(r.p99_abs),
                    fmt(r.max_rel),
                    fmt(r.p99_rel),
                    "" if r.cosine_similarity is None else f"{r.cosine_similarity:.9g}",
                    (r.tolerance or "").replace("|", "\\|"),
                    (r.note or "").replace("|", "\\|"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="all", help="all or comma-separated package names")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--backend", choices=["installed", "source"], default="installed")
    parser.add_argument("--vla-max-abs", type=float, default=0.03125)
    parser.add_argument("--vla-max-rel", type=float, default=0.05)
    parser.add_argument("--smallm-max-abs", type=float, default=2.0)
    parser.add_argument("--smallm-max-rel", type=float, default=0.02)
    parser.add_argument("--rel-floor", type=float, default=1.0)
    parser.add_argument("--bf16-max-ulp", type=int, default=1)
    parser.add_argument("--smallm-max-ulp", type=int, default=5)
    parser.add_argument("--smallm-chunk-rows", type=int, default=256)
    parser.add_argument("--output-json", default=None, help="optional path for a machine-readable correctness report")
    parser.add_argument("--output-md", default=None, help="optional path for a Markdown correctness report")
    parser.add_argument("--quiet", action="store_true", help="print only non-passing checks and the final summary")
    args = parser.parse_args()

    results: list[Result] = []
    sweepers = {
        "flashrt-gemm-epilogues": sweep_gemm_epilogues,
        "flashrt-fp8-ffn": sweep_fp8_ffn,
        "flashrt-vla-video": sweep_vla,
        "flashrt-nvfp4": sweep_nvfp4,
        "flashrt-smallm-gemm": sweep_smallm,
        "flashrt-fused-quant": sweep_fused_quant,
    }
    for package in _selected_packages(args.package):
        sweeper = sweepers.get(package)
        if sweeper is None:
            results.append(Result(package, "*", "*", "BLOCKED", note="unknown package"))
            continue
        sweeper(args, results)

    _print_results(results, quiet=args.quiet)
    if args.output_json:
        _write_json(ROOT / args.output_json, results, args)
    if args.output_md:
        _write_markdown(ROOT / args.output_md, results, args)
    bad = [r for r in results if r.status != "PASS"]
    if bad:
        print(f"accuracy sweep failed: {len(bad)} non-passing checks")
        return 1
    print(f"accuracy sweep passed: {len(results)} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
