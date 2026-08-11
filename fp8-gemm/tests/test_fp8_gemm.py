#!/usr/bin/env python3
"""Correctness tests for fp8-gemm."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "fp8-gemm"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


SHAPES = {
    "decode_m1_k512_n512": (1, 512, 512),
    "decode_m1_k4096_n2048": (1, 4096, 2048),
    "decode_m1_k4096_n8192": (1, 4096, 8192),
    "small_m8_k1024_n2048": (8, 1024, 2048),
    "small_m16_k4096_n4096": (16, 4096, 4096),
    "small_m32_k4096_n8192": (32, 4096, 8192),
    "small_m64_k512_n1024": (64, 512, 1024),
}

LARGE_M_SHAPES = {
    "large_m_boundary_65": (65, 2048, 2048),
    # PI0.5 / PI0 decoder and encoder projection families.
    "pi05_action_qkv": (51, 2048, 2560),
    "pi05_action_o": (51, 2048, 2048),
    "pi05_action_gate_up": (51, 2048, 16384),
    "pi05_action_down": (51, 8192, 2048),
    # GROOT N1.6/N1.7 DiT, backbone, and vision rows.
    "groot_dit_qkv": (51, 1536, 4608),
    "groot_n17_llm_o": (277, 2048, 2048),
    "groot_n17_llm_gate_up": (277, 2048, 16384),
    "groot_n17_llm_down": (277, 8192, 2048),
    "groot_n17_vit_o": (1024, 1024, 1024),
    # Cosmos Edge and LingBot projection families.
    "cosmos_edge_action": (64, 2048, 9216),
    "lingbot_vision_o": (1024, 1280, 1280),
    "lingbot_action_gate_up": (105, 2048, 16384),
    # PI0.5 Thor prefill tower, full real row envelope.
    "pi05_prefill_qkv": (712, 2048, 2560),
    "pi05_prefill_o": (970, 2048, 2048),
    "pi05_prefill_gate_up": (768, 2048, 32768),
    "pi05_prefill_down": (768, 16384, 2048),
}

MODES = {
    "smoke": ["decode_m1_k512_n512", "small_m8_k1024_n2048"],
    "headline": [
        "decode_m1_k4096_n2048",
        "decode_m1_k4096_n8192",
        "small_m16_k4096_n4096",
        "small_m32_k4096_n8192",
    ],
    "full": list(SHAPES.keys()),
}


@dataclass
class Metrics:
    shape: str
    M: int
    K: int
    N: int
    variant: int
    tile: str
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    dtype: str
    tolerance: str
    passed: bool


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    @staticmethod
    def select_fp8_linear_tile(m: int, n: int, k: int, variant: int = 0) -> str:
        return select_tile(m, n, k, variant)

    def fp8_linear_bf16(self, x, w, alpha=1.0, out=None, variant=0):
        if out is None:
            out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16)
        self._ops.fp8_linear_bf16(x, w, float(alpha), int(variant), out)
        return out

    def fp8_linear_residual_bf16(self, x, w, residual, alpha=1.0, variant=0):
        self._ops.fp8_linear_residual_bf16(x, w, float(alpha), int(variant), residual)
        return residual

    def fp8_linear_bias_bf16(self, x, w, bias, alpha=1.0, out=None):
        if out is None:
            out = torch.empty(
                (x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16
            )
        self._ops.fp8_linear_bias_bf16(x, w, bias, float(alpha), out)
        return out

    def fp8_linear_bias_residual_bf16(
        self, x, w, bias, residual, alpha=1.0
    ):
        self._ops.fp8_linear_bias_residual_bf16(
            x, w, bias, float(alpha), residual
        )
        return residual

    def fp8_linear_bias_gelu_bf16(self, x, w, bias, alpha=1.0, out=None):
        if out is None:
            out = torch.empty(
                (x.shape[0], w.shape[0]), device=x.device, dtype=torch.bfloat16
            )
        self._ops.fp8_linear_bias_gelu_bf16(x, w, bias, float(alpha), out)
        return out

    def fp8_blockwise_linear_bf16(
        self, x, w, input_scale, weight_scale, out=None
    ):
        if out is None:
            out = torch.empty(
                (x.shape[0], w.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
        self._ops.fp8_blockwise_linear_bf16(
            x, w, input_scale, weight_scale, out
        )
        return out

    def fp8_blockwise_swiglu_quantize_fp8(
        self, x, gate_up_weight, input_scale, gate_up_weight_scale,
        output=None, output_scale=None,
    ):
        n = gate_up_weight.shape[0] // 2
        if output is None:
            output = torch.empty(
                (x.shape[0], n), device=x.device, dtype=torch.float8_e4m3fn
            )
        if output_scale is None:
            output_scale = torch.empty(
                (x.shape[0], n // 128), device=x.device, dtype=torch.float32
            )
        self._ops.fp8_blockwise_swiglu_quantize_fp8(
            x, gate_up_weight, input_scale, gate_up_weight_scale,
            output, output_scale,
        )
        return output, output_scale


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) == (11, 0):
        return "11.0a"
    return "12.0a" if (major, minor) == (12, 0) else f"{major}.{minor}"


def load_source_wrapper(namespace: str):
    """Execute the shipped Python API against a source-test op namespace."""
    package_name = f"_{namespace}_public_api"
    ops_module = types.ModuleType(f"{package_name}._ops")
    ops_module.add_op_namespace_prefix = lambda name: f"{namespace}::{name}"
    ops_module.ops = getattr(torch.ops, namespace)
    sys.modules[ops_module.__name__] = ops_module

    init_path = PACKAGE / "torch-ext" / "fp8_gemm" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load public API from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def load_source_ops():
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "fp8_gemm_source_test"
    cutlass_include = Path(
        os.environ.get(
            "CUTLASS_INCLUDE",
            str(
                ROOT.parent
                / "flashrt_pr31_review"
                / "third_party"
                / "cutlass"
                / "include"
            ),
        )
    )
    if not (cutlass_include / "cutlass" / "cutlass.h").is_file():
        raise RuntimeError(
            "CUTLASS 4 include path is required; set CUTLASS_INCLUDE"
        )
    capability = torch.cuda.get_device_capability(0)
    if capability == (8, 9):
        cuda_sources = [
            str(PACKAGE / "csrc" / "fp8_block128_gemm_mma_sm89.cu"),
            str(PACKAGE / "csrc" / "fp8_gemv_m1_sm89.cu"),
        ]
        source_define = "-DFLASHRT_FP8_GEMM_SOURCE_SM89_ONLY"
    elif capability == (11, 0):
        cuda_sources = [
            str(PACKAGE / "csrc" / "cutlass_sm110_fp8_gemm.cu"),
            str(PACKAGE / "csrc" / "cublaslt_fp8_bias_sm110.cu"),
        ]
        source_define = "-DFLASHRT_FP8_GEMM_SOURCE_SM110_ONLY"
    else:
        cuda_sources = [
            str(PACKAGE / "csrc" / "fp8_gemv_m1_sm120.cu"),
            str(PACKAGE / "csrc" / "fp8_smallM_handtuned_sm120.cu"),
            str(PACKAGE / "csrc" / "fp8_smallM_handtuned_ldmatrix_sm120.cu"),
            str(PACKAGE / "csrc" / "cutlass_sm120_block128_fp8_gemm.cu"),
            str(PACKAGE / "csrc" / "cublaslt_fp8_bias_sm110.cu"),
        ]
        source_define = "-DFLASHRT_FP8_GEMM_SOURCE_SM120_ONLY"
    load(
        name=namespace,
        sources=[str(PACKAGE / "torch-ext" / "torch_binding.cpp"), *cuda_sources],
        extra_include_paths=[
            str(PACKAGE / "csrc"),
            str(REGISTRATION_INCLUDE),
            str(cutlass_include),
            str(cutlass_include.parent / "tools" / "util" / "include"),
        ],
        extra_cflags=["-O3", "-DNDEBUG", "-DCUDA_KERNEL", source_define],
        extra_cuda_cflags=[
            "-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-DCUDA_KERNEL", source_define
        ],
        verbose=False,
    )
    return load_source_wrapper(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("fp8_gemm")
    finally:
        if artifact:
            sys.path.remove(artifact)


def select_tile(m: int, n: int, k: int, variant: int = 0) -> str:
    if torch.cuda.get_device_capability(0) == (11, 0):
        forced = {1: "sm110_sq_bf16", 2: "sm110_t1_bf16", 3: "sm110_wide_bf16"}
        if variant not in {0, *forced}:
            raise RuntimeError("SM110 variant must be in [0, 3]")
        if variant:
            return forced[variant]
        if m >= 512 and k == 2048 and 2048 <= n <= 2560:
            return "sm110_sq_bf16"
        if m >= 512 and n >= 16 * k:
            return "sm110_t1_bf16"
        if m >= 512 and k >= 4 * n:
            return "sm110_wide_bf16"
        if n >= 8 * k:
            return "sm110_wide_bf16"
        if m >= 128 and k >= 4 * n:
            return "sm110_sq_bf16"
        if n == k and m >= 512:
            return "sm110_sq_bf16" if k <= 1024 else "sm110_wide_bf16"
        if n == k and m >= 128:
            return "sm110_wide_bf16"
        return "sm110_t1_bf16"
    if m == 1:
        if variant == 4:
            return "gemv_fp8_m1_w4"
        if variant == 8:
            return "gemv_fp8_m1_w8"
        if variant == 16:
            return "gemv_fp8_m1_w16"
        if n <= 2048:
            return "gemv_fp8_m1_w4"
        if n <= 8192:
            return "gemv_fp8_m1_w8"
        return "gemv_fp8_m1_w16"
    if m <= 16:
        if k % 256 == 0:
            return "ld_fp8_gemm_16x128x256_w4" if n % 128 == 0 else "ld_fp8_gemm_16x64x256_w4"
        if n % 256 == 0:
            return "ld_fp8_gemm_16x256x128_w8"
        if n % 192 == 0:
            return "ld_fp8_gemm_16x192x128_w4"
        if n % 128 == 0:
            return "ld_fp8_gemm_16x128x128_w4"
        return "ld_fp8_gemm_16x64x128_w4"
    if m <= 32:
        if k % 256 == 0:
            return "ld_fp8_gemm_32x128x256_w4" if n % 128 == 0 else "ld_fp8_gemm_32x64x256_w4"
        if n % 192 == 0:
            return "ld_fp8_gemm_32x192x128_w4"
        if n % 128 == 0:
            return "ld_fp8_gemm_32x128x128_w4"
        return "ld_fp8_gemm_32x64x128_w4"
    if m <= 64:
        if k % 256 == 0:
            return "ld_fp8_gemm_64x128x256_w4" if n % 128 == 0 else "ld_fp8_gemm_64x64x256_w4"
        if n % 128 == 0:
            return "ld_fp8_gemm_64x128x128_w4"
        return "ld_fp8_gemm_64x64x128_w4"
    if m <= 64:
        if k % 256 == 0:
            return "ld_fp8_gemm_64x128x256_w4" if n % 128 == 0 else "ld_fp8_gemm_64x64x256_w4"
        if n % 128 == 0:
            return "ld_fp8_gemm_64x128x128_w4"
        return "ld_fp8_gemm_64x64x128_w4"
    return "cublaslt_fp8_large_m"


def make_inputs(m: int, k: int, n: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x_bf16 = (torch.randn((m, k), device="cuda", generator=gen) * 0.25).to(torch.bfloat16)
    w_bf16 = (torch.randn((n, k), device="cuda", generator=gen) * 0.25).to(torch.bfloat16)
    x = x_bf16.to(torch.float8_e4m3fn)
    w = w_bf16.to(torch.float8_e4m3fn)
    return x, w


def reference(x: torch.Tensor, w: torch.Tensor, alpha: float) -> torch.Tensor:
    return ((x.float() @ w.float().T) * float(alpha)).to(torch.bfloat16)


def compare(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (got.float() - expected.float()).abs().flatten()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    p99_rank = max(1, min(diff.numel(), math.ceil(0.99 * diff.numel())))
    p99_abs = float(diff.kthvalue(p99_rank).values.item())
    cos = float(torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item())
    return max_abs, mean_abs, p99_abs, cos


def check_threshold(max_abs: float, mean_abs: float, p99_abs: float, cos: float) -> bool:
    return max_abs <= 0.5 and mean_abs <= 0.02 and p99_abs <= 0.25 and cos >= 0.999


def run_case(ops, name: str, shape: tuple[int, int, int], variant: int = 0) -> Metrics:
    m, k, n = shape
    x, w = make_inputs(m, k, n, seed=1000 + m + k + n + variant)
    alpha = 1.0
    expected = reference(x, w, alpha)
    got = ops.fp8_linear_bf16(x, w, alpha=alpha, variant=variant)
    torch.cuda.synchronize()
    max_abs, mean_abs, p99_abs, cos = compare(got, expected)
    tile = ops.select_fp8_linear_tile(m, n, k, variant)
    passed = check_threshold(max_abs, mean_abs, p99_abs, cos)
    return Metrics(
        shape=name,
        M=m,
        K=k,
        N=n,
        variant=variant,
        tile=tile,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine=cos,
        dtype=str(got.dtype),
        tolerance="max_abs<=0.5 mean_abs<=0.02 p99_abs<=0.25 cosine>=0.999",
        passed=passed,
    )


def run_residual_case(ops) -> Metrics:
    m, k, n = (1, 4096, 2048)
    x, w = make_inputs(m, k, n, seed=2026)
    residual = torch.randn((1, n), device="cuda", dtype=torch.bfloat16) * 0.1
    expected = (residual.float() + reference(x, w, 1.0).float()).to(torch.bfloat16)
    got = residual.clone()
    variant = 0 if torch.cuda.get_device_capability(0) == (11, 0) else 8
    ops.fp8_linear_residual_bf16(x, w, got, alpha=1.0, variant=variant)
    torch.cuda.synchronize()
    max_abs, mean_abs, p99_abs, cos = compare(got, expected)
    passed = check_threshold(max_abs, mean_abs, p99_abs, cos)
    return Metrics(
        shape="decode_residual_m1_k4096_n2048",
        M=m,
        K=k,
        N=n,
        variant=variant,
        tile=(
            "sm110_t1_bf16_residual"
            if torch.cuda.get_device_capability(0) == (11, 0)
            else "gemv_fp8_m1_resadd_w8"
        ),
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine=cos,
        dtype=str(got.dtype),
        tolerance="max_abs<=0.5 mean_abs<=0.02 p99_abs<=0.25 cosine>=0.999",
        passed=passed,
    )


def run_bias_cases(ops) -> int:
    count = 0
    shapes = [
        (512, 1152, 4304),
        (768, 4304, 1152),
        (768, 1152, 3456),
    ]
    for m, k, n in shapes:
        x, w = make_inputs(m, k, n, seed=7000 + m + k + n)
        bias = (torch.randn((n,), device="cuda") * 0.1).to(torch.bfloat16)
        alpha = 0.75
        base = (x.float() @ w.float().T) * alpha

        got = ops.fp8_linear_bias_bf16(x, w, bias, alpha=alpha)
        expected = (base + bias.float()).to(torch.bfloat16)
        maximum, mean, p99, cosine = compare(got, expected)
        assert maximum <= 0.5 and mean <= 0.02 and p99 <= 0.25 and cosine >= 0.999, (
            "bias", m, k, n, maximum, mean, p99, cosine
        )

        residual = (torch.randn((m, n), device="cuda") * 0.1).to(
            torch.bfloat16
        )
        residual_before = residual.clone()
        got_residual = ops.fp8_linear_bias_residual_bf16(
            x, w, bias, residual, alpha=alpha
        )
        expected_residual = (
            residual_before.float() + base + bias.float()
        ).to(torch.bfloat16)
        maximum, mean, p99, cosine = compare(got_residual, expected_residual)
        assert maximum <= 0.5 and mean <= 0.02 and p99 <= 0.25 and cosine >= 0.999, (
            "bias_residual", m, k, n, maximum, mean, p99, cosine
        )

        got_gelu = ops.fp8_linear_bias_gelu_bf16(x, w, bias, alpha=alpha)
        expected_gelu = torch.nn.functional.gelu(
            base + bias.float(), approximate="tanh"
        ).to(torch.bfloat16)
        maximum, mean, p99, cosine = compare(got_gelu, expected_gelu)
        assert maximum <= 0.5 and mean <= 0.02 and p99 <= 0.25 and cosine >= 0.999, (
            "bias_gelu", m, k, n, maximum, mean, p99, cosine
        )
        count += 3

    m, k, n = (512, 1152, 4304)
    x, w = make_inputs(m, k, n, seed=8801)
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)

    def invoke(input, weight, bias):
        return ops.fp8_linear_bias_bf16(input, weight, bias)

    eager = invoke(x, w, bias)
    compiled = torch.compile(invoke, fullgraph=True)(x, w, bias)
    torch.testing.assert_close(compiled, eager, rtol=0.0, atol=0.0)

    graph_out = torch.empty_like(eager)
    ops.fp8_linear_bias_bf16(x, w, bias, out=graph_out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.fp8_linear_bias_bf16(x, w, bias, out=graph_out)
    graph.replay()
    torch.testing.assert_close(graph_out, eager, rtol=0.0, atol=0.0)
    return count + 2


def run_blockwise_case(
    ops, name: str, shape: tuple[int, int, int]
) -> Metrics:
    m, k, n = shape
    gen = torch.Generator(device="cuda").manual_seed(5000 + m + k + n)
    x = (torch.randn((m, k), device="cuda", generator=gen) * 0.4).to(
        torch.float8_e4m3fn
    )
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.4).to(
        torch.float8_e4m3fn
    )
    input_scale = (
        0.005
        + 0.02
        * torch.rand((m, k // 128), device="cuda", generator=gen)
    ).float().contiguous()
    weight_scale = (
        0.005
        + 0.02
        * torch.rand((n // 128, k // 128), device="cuda", generator=gen)
    ).float().contiguous()
    expanded_input_scale = input_scale.repeat_interleave(128, dim=1)
    expanded_weight_scale = weight_scale.repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)
    expected = (
        (x.float() * expanded_input_scale)
        @ (w.float() * expanded_weight_scale).T
    ).to(torch.bfloat16)
    got = ops.fp8_blockwise_linear_bf16(
        x, w, input_scale, weight_scale
    )
    torch.cuda.synchronize()
    max_abs, mean_abs, p99_abs, cos = compare(got, expected)
    passed = (
        max_abs <= 0.0625
        and mean_abs <= 0.003
        and p99_abs <= 0.015625
        and cos >= 0.9999
    )
    return Metrics(
        shape=name,
        M=m,
        K=k,
        N=n,
        variant=0,
        tile=(
            "mma_sm89_block128"
            if torch.cuda.get_device_capability(0) == (8, 9)
            else "cutlass_sm120_block128"
        ),
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine=cos,
        dtype=str(got.dtype),
        tolerance=(
            "max_abs<=0.0625 mean_abs<=0.003 "
            "p99_abs<=0.015625 cosine>=0.9999"
        ),
        passed=passed,
    )


def run_blockwise_compile_case(ops) -> None:
    m, k, n = (51, 1536, 1536)
    gen = torch.Generator(device="cuda").manual_seed(9153)
    x = (torch.randn((m, k), device="cuda", generator=gen) * 0.4).to(
        torch.float8_e4m3fn
    )
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.4).to(
        torch.float8_e4m3fn
    )
    input_scale = torch.rand(
        (m, k // 128), device="cuda", generator=gen, dtype=torch.float32
    ).mul_(0.02).add_(0.005)
    weight_scale = torch.rand(
        (n // 128, k // 128),
        device="cuda",
        generator=gen,
        dtype=torch.float32,
    ).mul_(0.02).add_(0.005)

    def invoke(input, weight, input_scale, weight_scale):
        return ops.fp8_blockwise_linear_bf16(
            input, weight, input_scale, weight_scale
        )

    eager = invoke(x, w, input_scale, weight_scale)
    compiled = torch.compile(invoke, fullgraph=True)(
        x, w, input_scale, weight_scale
    )
    torch.testing.assert_close(compiled, eager, rtol=0.0, atol=0.0)


def run_sm89_swiglu_case(ops, m: int, n: int, k: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(8900 + m + n + k)
    x = (torch.randn((m, k), device="cuda", generator=gen) * 0.3).to(
        torch.float8_e4m3fn
    )
    weight = (
        torch.randn((2 * n, k), device="cuda", generator=gen) * 0.3
    ).to(torch.float8_e4m3fn)
    input_scale = torch.rand(
        (m, k // 128), device="cuda", generator=gen
    ).mul_(0.02).add_(0.005)
    weight_scale = torch.rand(
        (2 * n // 128, k // 128), device="cuda", generator=gen
    ).mul_(0.02).add_(0.005)
    output, output_scale = ops.fp8_blockwise_swiglu_quantize_fp8(
        x, weight, input_scale, weight_scale
    )
    expanded_x_scale = input_scale.repeat_interleave(128, dim=1)
    expanded_w_scale = weight_scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    x_f32 = x.float() * expanded_x_scale
    weight_f32 = weight.float() * expanded_w_scale
    gate, up = (x_f32 @ weight_f32.t()).split(n, dim=1)
    expected = (
        torch.nn.functional.silu(gate).bfloat16() * up.bfloat16()
    ).bfloat16()
    actual = (
        output.float() * output_scale.repeat_interleave(128, dim=1)
    ).bfloat16()
    maximum, mean, p99, cosine = compare(actual, expected)
    assert output.dtype == torch.float8_e4m3fn
    assert output_scale.dtype == torch.float32
    assert torch.isfinite(output_scale).all() and (output_scale > 0).all()
    assert cosine >= 0.999 and mean <= 0.01 and p99 <= 0.05, (
        m, n, k, maximum, mean, p99, cosine
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability(0)
    if capability not in {(8, 9), (11, 0), (12, 0)}:
        raise SystemExit(
            "fp8-gemm source tests require SM89, SM110, or SM120; "
            f"got SM{capability[0]}{capability[1]}"
        )

    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    rows = []
    if capability in {(11, 0), (12, 0)}:
        rows.extend(run_case(ops, name, SHAPES[name]) for name in MODES[args.mode])
        rows.append(run_residual_case(ops))
    if capability in {(11, 0), (12, 0)} and args.mode == "full":
        rows.extend(
            run_case(ops, name, shape) for name, shape in LARGE_M_SHAPES.items()
        )
        bias_count = run_bias_cases(ops)
    else:
        bias_count = 0
    if capability == (11, 0) and args.mode == "full":
        rows.extend(
            run_case(
                ops,
                f"sm110_forced_variant_{variant}",
                LARGE_M_SHAPES["pi05_action_gate_up"],
                variant,
            )
            for variant in (1, 2, 3)
        )
    if capability in {(8, 9), (12, 0)}:
        blockwise_shapes = [
            ("blockwise_decode", (1, 1024, 1024)),
            ("blockwise_action", (51, 1536, 1536)),
        ]
        if args.mode == "full":
            blockwise_shapes += [
                ("blockwise_groot", (277, 2048, 2048)),
                ("blockwise_vision", (1024, 1152, 1152)),
                ("blockwise_video", (2520, 3072, 3072)),
                ("blockwise_qwen_mlp", (128, 4096, 12288)),
            ]
        rows.extend(
            run_blockwise_case(ops, name, shape)
            for name, shape in blockwise_shapes
        )
        run_blockwise_compile_case(ops)
    if capability == (8, 9):
        for m, n, k in [
            (1, 128, 128), (16, 512, 1024), (31, 1536, 1536),
            (32, 2048, 4096), (51, 4096, 4096), (128, 4096, 4096),
            (256, 4096, 4096),
        ]:
            run_sm89_swiglu_case(ops, m, n, k)
        try:
            x = torch.zeros((257, 128), device="cuda", dtype=torch.float8_e4m3fn)
            w = torch.zeros((256, 128), device="cuda", dtype=torch.float8_e4m3fn)
            xs = torch.ones((257, 1), device="cuda", dtype=torch.float32)
            ws = torch.ones((2, 1), device="cuda", dtype=torch.float32)
            ops.fp8_blockwise_swiglu_quantize_fp8(x, w, xs, ws)
        except RuntimeError as error:
            assert "M <= 256" in str(error)
        else:
            raise AssertionError("M=257 must be rejected")

    failed = [row for row in rows if not row.passed]
    payload = {
        "passed": len(rows) - len(failed) + bias_count,
        "failed": len(failed),
        "rows": [asdict(row) for row in rows],
        "bias_checks": bias_count,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
