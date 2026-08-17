#!/usr/bin/env python3
"""Correctness tests for fp4-fused-ops."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "fp4-fused-ops"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)
DEFAULT_CUTLASS_INCLUDE = (
    ROOT.parent
    / "flashrt_pr31_review"
    / "third_party"
    / "cutlass"
    / "include"
)


SHAPES = {
    "tiny_rows1_dim1024": (1, 1024),
    "decode_rows10_dim2048": (10, 2048),
    "small_rows64_dim2048": (64, 2048),
    "prefill_rows128_dim4096": (128, 4096),
}

THOR_MODEL_SHAPES = {
    "pi05_action_rows51_dim2048": (51, 2048),
    "pi05_action_rows51_dim8192": (51, 8192),
    "groot_dit_rows51_dim1536": (51, 1536),
    "groot_backbone_rows277_dim2048": (277, 2048),
    "cosmos_edge_rows64_dim2048": (64, 2048),
    "lingbot_action_rows105_dim2048": (105, 2048),
}

MODES = {
    "smoke": ["tiny_rows1_dim1024", "decode_rows10_dim2048"],
    "full": list(SHAPES),
    "thor-models": list(THOR_MODEL_SHAPES),
}


@dataclass
class CaseResult:
    case: str
    rows: int
    dim: int
    check: str
    packed_equal: bool
    sfa_equal: bool
    residual_equal: bool | None
    max_abs: float | None
    mean_abs: float | None
    p99_abs: float | None
    cosine: float | None
    passed: bool


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)
        self._anchor = torch.empty((1,), device="cuda", dtype=torch.uint8)

    def sfa_size_bytes(self, rows: int, dim: int, is_sfb: bool = False) -> int:
        return int(self._ops.sfa_size_bytes_for(self._anchor, int(rows), int(dim), bool(is_sfb)))

    def alloc(self, rows: int, dim: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty((rows, dim // 2), device=device, dtype=torch.uint8),
            torch.zeros((self.sfa_size_bytes(rows, dim, False),), device=device, dtype=torch.uint8),
        )

    def silu_mul_quantize_fp4_sfa_bf16(self, merged, packed, sfa):
        self._ops.silu_mul_quantize_fp4_sfa_bf16(merged, packed, sfa)

    def rms_norm_quantize_fp4_sfa_bf16(self, x, weight, eps, normed, packed, sfa):
        self._ops.rms_norm_quantize_fp4_sfa_bf16(
            x, weight, float(eps), normed, packed, sfa
        )

    def rms_norm_fp4_sfa_fp16(self, x, packed, sfa):
        self._ops.rms_norm_fp4_sfa_fp16(x, packed, sfa)

    def residual_add_rms_norm_fp4_sfa_fp16(self, residual, x, packed, sfa):
        self._ops.residual_add_rms_norm_fp4_sfa_fp16(residual, x, packed, sfa)

    def residual_add_rms_norm_fp4_sfa_v2_fp16(self, residual, x, packed, sfa):
        self._ops.residual_add_rms_norm_fp4_sfa_v2_fp16(residual, x, packed, sfa)

    def residual_add_rms_norm_mul_fp4_sfa_fp16(self, residual, x, inv_s, packed, sfa):
        self._ops.residual_add_rms_norm_mul_fp4_sfa_fp16(residual, x, inv_s, packed, sfa)

    def silu_mul_fp4_sfa_fp16(self, merged, packed, sfa):
        self._ops.silu_mul_fp4_sfa_fp16(merged, packed, sfa)

    def silu_mul_fp4_sfa_v2_fp16(self, merged, packed, sfa):
        self._ops.silu_mul_fp4_sfa_v2_fp16(merged, packed, sfa)

    def silu_mul_mul_fp4_sfa_v2_fp16(self, merged, inv_s, packed, sfa):
        self._ops.silu_mul_mul_fp4_sfa_v2_fp16(merged, inv_s, packed, sfa)

    def silu_mul_two_fp4_to_fp4(self, gate_packed, gate_sfa, up_packed, up_sfa, out_packed, out_sfa):
        self._ops.silu_mul_two_fp4_to_fp4(gate_packed, gate_sfa, up_packed, up_sfa, out_packed, out_sfa)

    def silu_mul_two_mul_fp4_to_fp4(self, gate_packed, gate_sfa, up_packed, up_sfa, inv_s, out_packed, out_sfa):
        self._ops.silu_mul_two_mul_fp4_to_fp4(
            gate_packed, gate_sfa, up_packed, up_sfa, inv_s, out_packed, out_sfa
        )

    def geglu_two_mul_nvfp4_native(self, gate_packed, gate_sfa, up_packed, up_sfa, inv_s, out_packed, out_sfa):
        self._ops.geglu_two_mul_nvfp4_native(
            gate_packed, gate_sfa, up_packed, up_sfa, inv_s, out_packed, out_sfa
        )

    def gelu_mul_nvfp4_bf16(self, merged, inv_s, packed, sfa):
        self._ops.gelu_mul_nvfp4_bf16(merged, inv_s, packed, sfa)

    def rms_norm_mul_nvfp4_bf16(self, x, inv_s, eps, packed, sfa):
        self._ops.rms_norm_mul_nvfp4_bf16(x, inv_s, float(eps), packed, sfa)

    def residual_add_rms_norm_nvfp4_bf16(self, residual, x, inv_s, eps, packed, sfa):
        self._ops.residual_add_rms_norm_nvfp4_bf16(
            residual, x, inv_s, float(eps), packed, sfa
        )

    def layer_norm_fp8_bf16(self, x, gamma, beta, eps, out):
        self._ops.layer_norm_fp8_bf16(x, gamma, beta, float(eps), out)

    def layer_norm_nvfp4_bf16(self, x, gamma, beta, inv_s, eps, packed, sfa):
        self._ops.layer_norm_nvfp4_bf16(
            x, gamma, beta, inv_s, float(eps), packed, sfa
        )

    def adaptive_rms_norm_nvfp4_fp16(self, x, style, packed, sfa, gate):
        self._ops.adaptive_rms_norm_nvfp4_fp16(x, style, packed, sfa, gate)

    def gated_residual_adaptive_rms_norm_nvfp4_fp16(
        self, x, previous_gate, residual, style, packed, sfa, gate
    ):
        self._ops.gated_residual_adaptive_rms_norm_nvfp4_fp16(
            x, previous_gate, residual, style, packed, sfa, gate
        )

    def adaptive_rms_norm_nvfp4_bf16(self, x, style, packed, sfa, gate):
        self._ops.adaptive_rms_norm_nvfp4_bf16(x, style, packed, sfa, gate)

    def gated_residual_adaptive_rms_norm_nvfp4_bf16(
        self, x, previous_gate, residual, style, packed, sfa, gate
    ):
        self._ops.gated_residual_adaptive_rms_norm_nvfp4_bf16(
            x, previous_gate, residual, style, packed, sfa, gate
        )

    def adaptive_rms_norm_fp8_static_fp16(self, x, style, scale, out, gate):
        self._ops.adaptive_rms_norm_fp8_static_fp16(x, style, scale, out, gate)

    def gated_residual_adaptive_rms_norm_fp8_static_fp16(
        self, x, previous_gate, residual, style, scale, out, gate
    ):
        self._ops.gated_residual_adaptive_rms_norm_fp8_static_fp16(
            x, previous_gate, residual, style, scale, out, gate
        )

    def adaptive_rms_norm_e0m3_fp16(self, x, style, use_rht, packed, sfa, gate):
        self._ops.adaptive_rms_norm_e0m3_fp16(
            x, style, bool(use_rht), packed, sfa, gate
        )

    def gated_residual_adaptive_rms_norm_e0m3_fp16(
        self, x, previous_gate, residual, style, use_rht, packed, sfa, gate
    ):
        self._ops.gated_residual_adaptive_rms_norm_e0m3_fp16(
            x, previous_gate, residual, style, bool(use_rht), packed, sfa, gate
        )

    def gelu_mul_e0m3_fp16(self, merged, use_rht, packed, sfa):
        self._ops.gelu_mul_e0m3_fp16(
            merged, bool(use_rht), packed, sfa
        )

    def residual_add_rms_norm_quant_nvfp4_swizzled_bf16(
        self, residual, x, weight, eps, packed, sfa
    ):
        self._ops.residual_add_rms_norm_quant_nvfp4_swizzled_bf16(
            residual, x, weight, float(eps), packed, sfa
        )

    def relu2_quant_nvfp4_swizzled_fp16(self, x, packed, sfa):
        self._ops.relu2_quant_nvfp4_swizzled_fp16(x, packed, sfa)

    def layer_norm_fp8_fp16(self, x, gamma, beta, eps, out):
        self._ops.layer_norm_fp8_fp16(x, gamma, beta, float(eps), out)

    def layer_norm_nvfp4_fp16(
        self, x, gamma, beta, inv_s, eps, packed, sfa
    ):
        self._ops.layer_norm_nvfp4_fp16(
            x, gamma, beta, inv_s, float(eps), packed, sfa
        )

    def gelu_mul_nvfp4_fp16(self, merged, packed, sfa):
        self._ops.gelu_mul_nvfp4_fp16(merged, packed, sfa)

    def dequantize_fp4_sfa_fp16(self, packed, sfa, out):
        self._ops.dequantize_fp4_sfa_fp16(packed, sfa, out)

    def rms_silu_nvfp4_ndhwc_bf16(
        self, x, gamma, awq_inv_scale=None, eps=1e-6
    ):
        b, c, t, h, w = x.shape
        packed = torch.empty(
            (b, t, h, w, c // 2), device=x.device, dtype=torch.uint8
        )
        scale_factors = torch.empty(
            (b, t, h, w, c // 16), device=x.device, dtype=torch.uint8
        )
        self._ops.rms_silu_nvfp4_ndhwc_bf16(
            x, gamma, awq_inv_scale, float(eps), packed, scale_factors
        )
        return packed, scale_factors

    def quantize_bf16_to_nvfp4_linear(self, input):
        rows, cols = input.shape
        packed = torch.empty(
            (rows, cols // 2), device=input.device, dtype=torch.uint8
        )
        scale_factors = torch.empty(
            (rows, cols // 16), device=input.device, dtype=torch.uint8
        )
        self._ops.quantize_bf16_to_nvfp4_linear(
            input, packed, scale_factors
        )
        return packed, scale_factors

    def bf16_rms_silu_ncdhw(
        self, x, gamma, prev_cache=None, eps=1e-6, out=None, next_cache=None
    ):
        out = torch.empty_like(x) if out is None else out
        self._ops.bf16_rms_silu_ncdhw(
            x, gamma, prev_cache, float(eps), out, next_cache
        )
        return out, next_cache

    def bf16_rms_norm_ncdhw(self, x, gamma, bias=None, eps=1e-6, out=None):
        out = torch.empty_like(x) if out is None else out
        self._ops.bf16_rms_norm_ncdhw(x, gamma, bias, float(eps), out)
        return out


def alloc_fp4(ops, rows: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate buffers through the public SFA sizing contract."""

    return (
        torch.empty((rows, dim // 2), device="cuda", dtype=torch.uint8),
        torch.zeros(
            (ops.sfa_size_bytes(rows, dim, False),),
            device="cuda",
            dtype=torch.uint8,
        ),
    )


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) == (11, 0):
        return "11.0a"
    if major >= 12:
        return "12.0a"
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    cutlass_include = Path(os.environ.get("FLASHRT_CUTLASS_INCLUDE", str(DEFAULT_CUTLASS_INCLUDE)))
    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    if not cutlass_include.is_dir():
        raise RuntimeError(f"missing CUTLASS include path: {cutlass_include}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "fp4_fused_ops_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "kernels" / "silu_mul_quantize_fp4_sfa_bf16.cu"),
            str(PACKAGE / "csrc" / "kernels" / "rms_norm_quantize_fp4_sfa_bf16.cu"),
            str(PACKAGE / "tests" / "request2_reference_binding.cpp"),
            str(ROOT / "fp4-gemm" / "csrc" / "quantize" / "quantize_fp4_sfa_bf16.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "norm_silu_fp4_sfa.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "adarms_nvfp4_bf16.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "pi05_bf16_fp4_producers.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "layer_norm_fp4_sfa.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "siglip_ln_vec.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "silu_mul_fp4_sfa_vec.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "dequantize_fp4_sfa.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "adarms_fp8_static_fp16.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "pi05_e0m3_act.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "cosmos3_edge_fp4.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "res_rms_fp4_sfa_v2.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "res_rms_mul_fp4_sfa.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "silu_mul_fp4_sfa_v2.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "silu_mul_mul_fp4_sfa_v2.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "silu_mul_two_fp4_to_fp4.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "rms_silu_nvfp4_ndhwc_bf16.cu"),
            str(PACKAGE / "csrc" / "quantize" / "quantize_bf16_to_nvfp4_linear.cu"),
            str(PACKAGE / "csrc" / "quantize" / "bf16_rms_silu_ncdhw.cu"),
            str(PACKAGE / "csrc" / "quantize" / "reshape_scales_sfa.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(PACKAGE / "csrc" / "quantize"), str(ROOT / "fp4-gemm" / "csrc"), str(ROOT / "fp4-gemm" / "csrc" / "quantize"), str(cutlass_include), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-std=c++17",
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-DCUDA_KERNEL",
            "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
            "-DFLASHRT_HAVE_COSMOS3_EDGE=1",
        ],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        module = importlib.import_module("fp4_fused_ops")
        # Exercise the installed torch.library ABI with the same explicit
        # output-buffer contract used by source tests.  The public Python
        # helpers intentionally expose a more convenient keyword API.
        adapter = object.__new__(SourceOps)
        adapter._ops = module.ops
        adapter._anchor = torch.empty((1,), device="cuda", dtype=torch.uint8)
        return adapter
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_fp16(shape: tuple[int, int], seed: int, scale: float = 0.25) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=gen) * scale).to(torch.float16).contiguous()


def make_bf16(shape: tuple[int, int], seed: int, scale: float = 0.25) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=gen) * scale).to(torch.bfloat16).contiguous()


def check_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    torch.cuda.synchronize()
    return bool(torch.equal(a, b))


def dequant_metrics(ops: SourceOps, packed_a, sfa_a, packed_b, sfa_b) -> tuple[float, float, float, float]:
    out_a = torch.empty((packed_a.shape[0], packed_a.shape[1] * 2), device=packed_a.device, dtype=torch.float16)
    out_b = torch.empty_like(out_a)
    ops.dequantize_fp4_sfa_fp16(packed_a, sfa_a, out_a)
    ops.dequantize_fp4_sfa_fp16(packed_b, sfa_b, out_b)
    torch.cuda.synchronize()
    diff = (out_a.float() - out_b.float()).abs().flatten()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    p99_abs = float(torch.quantile(diff, 0.99).item())
    cosine = float(torch.nn.functional.cosine_similarity(out_a.float().flatten(), out_b.float().flatten(), dim=0).item())
    return max_abs, mean_abs, p99_abs, cosine


def dequant_metrics_vs_ref(ops: SourceOps, packed, sfa, ref: torch.Tensor) -> tuple[float, float, float, float]:
    out = torch.empty((packed.shape[0], packed.shape[1] * 2), device=packed.device, dtype=torch.float16)
    ops.dequantize_fp4_sfa_fp16(packed, sfa, out)
    torch.cuda.synchronize()
    diff = (out.float() - ref.float()).abs().flatten()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    p99_abs = float(torch.quantile(diff, 0.99).item())
    cosine = float(torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item())
    return max_abs, mean_abs, p99_abs, cosine


def check_fp4_path_equivalence_threshold(max_abs: float, mean_abs: float, p99_abs: float, cosine: float) -> bool:
    return max_abs <= 1.0 and mean_abs <= 0.002 and p99_abs <= 0.03125 and cosine >= 0.99985


def check_fp4_quant_reference_threshold(max_abs: float, mean_abs: float, p99_abs: float, cosine: float) -> bool:
    return max_abs <= 1.0 and mean_abs <= 0.10 and p99_abs <= 0.40 and cosine >= 0.99


def _reference_nvfp4_sfa(x: torch.Tensor, sfa_bytes: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows, dim = x.shape
    vals = x.float().view(rows, dim // 16, 16)
    desired = (vals.abs().amax(dim=2) / 6.0).clamp_min(1e-12)
    scale_fp8 = desired.to(torch.float8_e4m3fn)
    scaled = vals / scale_fp8.float().unsqueeze(-1)
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=x.device,
    )
    code = (scaled.abs().unsqueeze(-1) > thresholds).sum(dim=-1).to(torch.uint8)
    code |= (scaled < 0).to(torch.uint8) << 3
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).reshape(rows, dim // 2)
    sfa = torch.zeros((sfa_bytes,), device=x.device, dtype=torch.uint8)
    row = torch.arange(rows, device=x.device).view(-1, 1)
    block = torch.arange(dim // 16, device=x.device).view(1, -1)
    offset = (
        (row >> 7) * ((dim + 63) // 64) * 512
        + (block >> 2) * 512
        + (row & 31) * 16
        + (row >> 5 & 3) * 4
        + (block & 3)
    )
    sfa[offset.flatten()] = scale_fp8.view(torch.uint8).flatten()
    return packed, sfa


def run_request2_checks(ops: SourceOps) -> None:
    for rows in (1, 7, 257):
        hidden = 1024
        merged = make_bf16((rows, 2 * hidden), 2026081700 + rows, 0.2)
        packed, sfa = ops.alloc(rows, hidden)
        ops.silu_mul_quantize_fp4_sfa_bf16(merged, packed, sfa)
        staged = (
            torch.nn.functional.silu(merged[:, :hidden].float())
            * merged[:, hidden:].float()
        ).bfloat16()
        ref_packed, ref_sfa = ops.alloc(rows, hidden)
        torch.ops.fp4_request2_reference.quantize(staged, ref_packed, ref_sfa)
        if not torch.equal(packed, ref_packed) or not torch.equal(sfa, ref_sfa):
            raise AssertionError(f"silu+NVFP4 production-chain mismatch rows={rows}")

    for rows, dim in ((1, 2048), (63, 2048), (257, 4096)):
        x = make_bf16((rows, dim), 2026081800 + rows, 0.2)
        weight = make_bf16((1, dim), 2026081900 + rows, 0.05).view(dim)
        normed = torch.empty_like(x)
        packed, sfa = ops.alloc(rows, dim)
        ops.rms_norm_quantize_fp4_sfa_bf16(
            x, weight, 1e-6, normed, packed, sfa
        )
        expected = (
            x.float()
            * torch.rsqrt(x.float().square().mean(dim=1, keepdim=True) + 1e-6)
            * (1.0 + weight.float())
        ).bfloat16()
        max_abs = float((normed.float() - expected.float()).abs().max().item())
        cosine = float(torch.nn.functional.cosine_similarity(
            normed.float().flatten(), expected.float().flatten(), dim=0
        ).item())
        if max_abs > 0.015625 or cosine < 0.99999:
            raise AssertionError(
                f"RMSNorm rows={rows} dim={dim}: max_abs={max_abs} cosine={cosine}"
            )
        ref_packed, ref_sfa = ops.alloc(rows, dim)
        torch.ops.fp4_request2_reference.quantize(normed, ref_packed, ref_sfa)
        if not torch.equal(packed, ref_packed) or not torch.equal(sfa, ref_sfa):
            raise AssertionError(f"RMSNorm production quant mismatch rows={rows} dim={dim}")

    merged = make_bf16((7, 2048), 2026082001, 0.2)
    packed, sfa = ops.alloc(7, 1024)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.silu_mul_quantize_fp4_sfa_bf16(merged, packed, sfa)
    graph.replay()
    torch.cuda.synchronize()
    expected_packed, expected_sfa = packed.clone(), sfa.clone()
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(packed, expected_packed) or not torch.equal(sfa, expected_sfa):
        raise AssertionError("request-2 FP4 producer graph replay is not bitwise stable")


def run_fp8_adarms_checks(ops) -> list[CaseResult]:
    results: list[CaseResult] = []
    rows, dim = 10, 1024
    x = make_fp16((rows, dim), 2026080601, 0.2)
    style = make_fp16((rows, 3 * dim), 2026080602, 0.1)
    scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    gate = torch.empty_like(x)
    ops.adaptive_rms_norm_fp8_static_fp16(x, style, scale, out, gate)
    torch.cuda.synchronize()
    mod_scale, shift, gate_ref = style.chunk(3, dim=-1)
    rstd = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = x.float() * rstd * (1.0 + mod_scale.float()) + shift.float()
    expected_fp8 = (expected / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequant = out.float() * scale
    diff = (dequant - expected).abs().flatten()
    cosine = float(torch.nn.functional.cosine_similarity(
        dequant.flatten(), expected.flatten(), dim=0).item())
    metrics = (float(diff.max()), float(diff.mean()), float(torch.quantile(diff, 0.99)), cosine)
    gate_equal = torch.equal(gate, gate_ref)
    results.append(CaseResult(
        case="pi05_adaptive_rms_fp8_rows10_dim1024", rows=rows, dim=dim,
        check="adaptive_rms_norm_fp8_bit_exact_math", packed_equal=torch.equal(out.view(torch.uint8), expected_fp8.view(torch.uint8)),
        sfa_equal=True, residual_equal=gate_equal,
        max_abs=metrics[0], mean_abs=metrics[1], p99_abs=metrics[2],
        cosine=metrics[3], passed=(gate_equal and torch.equal(out.view(torch.uint8), expected_fp8.view(torch.uint8))),
    ))

    delta = make_fp16((rows, dim), 2026080603, 0.1)
    previous_gate = make_fp16((rows, dim), 2026080604, 0.1)
    residual = make_fp16((rows, dim), 2026080605, 0.15)
    residual_initial = residual.clone()
    residual_math = residual.float() + delta.float() * previous_gate.float()
    residual_ref = residual_math.to(torch.float16)
    ops.gated_residual_adaptive_rms_norm_fp8_static_fp16(
        delta, previous_gate, residual, style, scale, out, gate
    )
    torch.cuda.synchronize()
    rstd = torch.rsqrt(residual_math.square().mean(-1, keepdim=True) + 1e-6)
    expected = residual_ref.float() * rstd * (1.0 + mod_scale.float()) + shift.float()
    expected_fp8 = (expected / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequant = out.float() * scale
    diff = (dequant - expected).abs().flatten()
    cosine = float(torch.nn.functional.cosine_similarity(
        dequant.flatten(), expected.flatten(), dim=0).item())
    metrics = (float(diff.max()), float(diff.mean()), float(torch.quantile(diff, 0.99)), cosine)
    residual_equal = torch.equal(residual, residual_ref)
    graph = torch.cuda.CUDAGraph()
    residual.copy_(residual_initial)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        ops.gated_residual_adaptive_rms_norm_fp8_static_fp16(
            delta, previous_gate, residual, style, scale, out, gate
        )
    residual.copy_(residual_initial)
    graph.replay()
    torch.cuda.synchronize()
    first = (residual.clone(), out.clone(), gate.clone())
    residual.copy_(residual_initial)
    graph.replay()
    torch.cuda.synchronize()
    second = (residual.clone(), out.clone(), gate.clone())
    graph_equal = all(torch.equal(a, b) for a, b in zip(first, second))
    results.append(CaseResult(
        case="pi05_gate_res_adaptive_rms_fp8_rows10_dim1024", rows=rows, dim=dim,
        check="gate_res_adaptive_rms_fp8_bit_exact_math_graph", packed_equal=(graph_equal and torch.equal(out.view(torch.uint8), expected_fp8.view(torch.uint8))),
        sfa_equal=True, residual_equal=residual_equal,
        max_abs=metrics[0], mean_abs=metrics[1], p99_abs=metrics[2],
        cosine=metrics[3], passed=(residual_equal and graph_equal and torch.equal(out.view(torch.uint8), expected_fp8.view(torch.uint8))),
    ))
    return results


def run_bf16_adarms_nvfp4_checks(ops) -> list[CaseResult]:
    results: list[CaseResult] = []
    dim = 1024
    for rows in (1, 10, 51, 105):
        x = make_bf16((rows, dim), 2026080700 + rows, 0.2)
        style = make_bf16((rows, 3 * dim), 2026080800 + rows, 0.1)
        packed, sfa = alloc_fp4(ops, rows, dim)
        gate = torch.empty_like(x)
        ops.adaptive_rms_norm_nvfp4_bf16(x, style, packed, sfa, gate)
        torch.cuda.synchronize()

        scale, shift, gate_ref = style.chunk(3, dim=-1)
        rstd = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        norm_ref = (
            x.float() * rstd * (1.0 + scale.float()) + shift.float()
        ).to(torch.bfloat16)
        max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
            ops, packed, sfa, norm_ref.half()
        )
        gate_exact = torch.equal(gate, gate_ref)
        results.append(CaseResult(
            case=f"adarms_nvfp4_bf16_rows{rows}_dim{dim}",
            rows=rows, dim=dim, check="gate_exact_and_nvfp4_vs_bf16_math",
            packed_equal=True, sfa_equal=True, residual_equal=gate_exact,
            max_abs=max_abs, mean_abs=mean_abs, p99_abs=p99_abs,
            cosine=cosine,
            passed=(gate_exact and check_fp4_quant_reference_threshold(
                max_abs, mean_abs, p99_abs, cosine
            )),
        ))

        previous_gate = make_bf16((rows, dim), 2026080900 + rows, 0.1)
        residual = make_bf16((rows, dim), 2026081000 + rows, 0.15)
        residual_initial = residual.clone()
        update_math = residual.float() + x.float() * previous_gate.float()
        residual_ref = update_math.to(torch.bfloat16)
        ops.gated_residual_adaptive_rms_norm_nvfp4_bf16(
            x, previous_gate, residual, style, packed, sfa, gate
        )
        torch.cuda.synchronize()
        rstd = torch.rsqrt(update_math.square().mean(-1, keepdim=True) + 1e-6)
        norm_ref = (
            residual_ref.float() * rstd * (1.0 + scale.float()) + shift.float()
        ).to(torch.bfloat16)
        max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
            ops, packed, sfa, norm_ref.half()
        )
        residual_exact = torch.equal(residual, residual_ref)
        gate_exact = torch.equal(gate, gate_ref)

        graph_exact = True
        if rows == 10:
            graph = torch.cuda.CUDAGraph()
            residual.copy_(residual_initial)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                ops.gated_residual_adaptive_rms_norm_nvfp4_bf16(
                    x, previous_gate, residual, style, packed, sfa, gate
                )
            residual.copy_(residual_initial)
            graph.replay()
            torch.cuda.synchronize()
            first = (residual.clone(), packed.clone(), sfa.clone(), gate.clone())
            residual.copy_(residual_initial)
            graph.replay()
            torch.cuda.synchronize()
            second = (residual.clone(), packed.clone(), sfa.clone(), gate.clone())
            graph_exact = all(torch.equal(a, b) for a, b in zip(first, second))

        results.append(CaseResult(
            case=f"gate_res_adarms_nvfp4_bf16_rows{rows}_dim{dim}",
            rows=rows, dim=dim,
            check="residual_gate_graph_exact_and_nvfp4_vs_bf16_math",
            packed_equal=graph_exact, sfa_equal=graph_exact,
            residual_equal=residual_exact,
            max_abs=max_abs, mean_abs=mean_abs, p99_abs=p99_abs,
            cosine=cosine,
            passed=(residual_exact and gate_exact and graph_exact and
                    check_fp4_quant_reference_threshold(
                        max_abs, mean_abs, p99_abs, cosine
                    )),
        ))
    return results


def run_e0m3_and_cosmos_fp4_checks(ops) -> list[CaseResult]:
    results: list[CaseResult] = []
    rows, dim = 10, 1024
    x = make_fp16((rows, dim), 2026080611, 0.2)
    style = make_fp16((rows, 3 * dim), 2026080612, 0.1)
    packed_a, sfa_a = alloc_fp4(ops, rows, dim)
    packed_b, sfa_b = alloc_fp4(ops, rows, dim)
    gate_a, gate_b = torch.empty_like(x), torch.empty_like(x)
    ops.adaptive_rms_norm_e0m3_fp16(
        x, style, False, packed_a, sfa_a, gate_a
    )
    ops.adaptive_rms_norm_e0m3_fp16(
        x, style, False, packed_b, sfa_b, gate_b
    )
    torch.cuda.synchronize()
    deterministic = (
        torch.equal(packed_a, packed_b)
        and torch.equal(sfa_a, sfa_b)
        and torch.equal(gate_a, gate_b)
    )
    gate_exact = torch.equal(gate_a, style[:, 2 * dim :])
    results.append(CaseResult(
        case="pi05_adaptive_rms_e0m3_rows10_dim1024", rows=rows, dim=dim,
        check="e0m3_deterministic_and_gate_exact", packed_equal=deterministic,
        sfa_equal=torch.equal(sfa_a, sfa_b), residual_equal=gate_exact,
        max_abs=None, mean_abs=None, p99_abs=None, cosine=None,
        passed=(deterministic and gate_exact and int(packed_a.sum()) != 0),
    ))

    previous_gate = make_fp16((rows, dim), 2026080613, 0.1)
    residual = make_fp16((rows, dim), 2026080614, 0.15)
    residual_initial = residual.clone()
    residual_ref = (
        residual_initial.float() + x.float() * previous_gate.float()
    ).half()
    ops.gated_residual_adaptive_rms_norm_e0m3_fp16(
        x, previous_gate, residual, style, True, packed_a, sfa_a, gate_a
    )
    torch.cuda.synchronize()
    residual_exact = torch.equal(residual, residual_ref)
    graph = torch.cuda.CUDAGraph()
    residual.copy_(residual_initial)
    with torch.cuda.graph(graph):
        ops.gated_residual_adaptive_rms_norm_e0m3_fp16(
            x, previous_gate, residual, style, True, packed_a, sfa_a, gate_a
        )
    residual.copy_(residual_initial)
    graph.replay()
    first = (residual.clone(), packed_a.clone(), sfa_a.clone(), gate_a.clone())
    residual.copy_(residual_initial)
    graph.replay()
    torch.cuda.synchronize()
    graph_exact = all(
        torch.equal(a, b)
        for a, b in zip(first, (residual, packed_a, sfa_a, gate_a))
    )
    results.append(CaseResult(
        case="pi05_gate_res_adaptive_rms_e0m3_rht_rows10_dim1024",
        rows=rows, dim=dim, check="residual_and_graph_bit_exact",
        packed_equal=graph_exact, sfa_equal=graph_exact,
        residual_equal=residual_exact, max_abs=None, mean_abs=None,
        p99_abs=None, cosine=None,
        passed=(residual_exact and graph_exact and torch.equal(gate_a, style[:, 2 * dim :])),
    ))

    merged = make_fp16((rows, 2 * 2048), 2026080615, 0.2)
    geglu_a, geglu_sfa_a = alloc_fp4(ops, rows, 2048)
    geglu_b, geglu_sfa_b = alloc_fp4(ops, rows, 2048)
    ops.gelu_mul_e0m3_fp16(merged, True, geglu_a, geglu_sfa_a)
    ops.gelu_mul_e0m3_fp16(merged, True, geglu_b, geglu_sfa_b)
    torch.cuda.synchronize()
    geglu_exact = torch.equal(geglu_a, geglu_b) and torch.equal(geglu_sfa_a, geglu_sfa_b)
    results.append(CaseResult(
        case="pi05_gelu_mul_e0m3_rht_rows10_dim2048", rows=rows, dim=2048,
        check="e0m3_geglu_deterministic", packed_equal=geglu_exact,
        sfa_equal=torch.equal(geglu_sfa_a, geglu_sfa_b), residual_equal=None,
        max_abs=None, mean_abs=None, p99_abs=None, cosine=None,
        passed=(geglu_exact and int(geglu_a.sum()) != 0),
    ))

    rows, dim = 51, 2048
    residual = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16) * 0.1
    update = torch.randn_like(residual) * 0.1
    weight = torch.randn((dim,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1
    residual_initial = residual.clone()
    packed, sfa = alloc_fp4(ops, rows, dim)
    ops.residual_add_rms_norm_quant_nvfp4_swizzled_bf16(
        residual, update, weight, 1e-6, packed, sfa
    )
    torch.cuda.synchronize()
    residual_math = residual_initial.float() + update.float()
    residual_ref = residual_math.bfloat16()
    norm_ref = residual_math * torch.rsqrt(
        residual_math.square().mean(-1, keepdim=True) + 1e-6
    ) * weight.float()
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
        ops, packed, sfa, norm_ref.half()
    )
    residual_exact = torch.equal(residual, residual_ref)
    results.append(CaseResult(
        case="cosmos_edge_res_rms_nvfp4_rows51_dim2048", rows=rows, dim=dim,
        check="residual_exact_and_nvfp4_vs_math", packed_equal=True,
        sfa_equal=True, residual_equal=residual_exact, max_abs=max_abs,
        mean_abs=mean_abs, p99_abs=p99_abs, cosine=cosine,
        passed=(residual_exact and check_fp4_quant_reference_threshold(max_abs, mean_abs, p99_abs, cosine)),
    ))

    relu_input = torch.randn((rows, 8192), device="cuda", dtype=torch.float16) * 0.2
    packed, sfa = alloc_fp4(ops, rows, 8192)
    ops.relu2_quant_nvfp4_swizzled_fp16(relu_input, packed, sfa)
    torch.cuda.synchronize()
    relu_ref = torch.relu(relu_input.float()).square().half()
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
        ops, packed, sfa, relu_ref
    )
    results.append(CaseResult(
        case="cosmos_edge_relu2_nvfp4_rows51_dim8192", rows=rows, dim=8192,
        check="relu2_nvfp4_vs_math", packed_equal=True, sfa_equal=True,
        residual_equal=None, max_abs=max_abs, mean_abs=mean_abs,
        p99_abs=p99_abs, cosine=cosine,
        passed=check_fp4_quant_reference_threshold(max_abs, mean_abs, p99_abs, cosine),
    ))
    return results


def run_pi05_thor_producer_checks(ops) -> list[CaseResult]:
    """Validate SM110 producers and report their expected quantization loss.

    These kernels emit NVFP4/FP8, so comparing their dequantized values with
    the pre-quantization FP16 tensor is a quantization-quality check, not an
    exact-equivalence check.  Residual and gate outputs remain bit-exact.  The
    native producer implementations are copied byte-for-byte from the PI0.5
    production runtime and are additionally covered there by fused-vs-staged
    and end-to-end parity tests.
    """

    if torch.cuda.get_device_capability(0) != (11, 0):
        return []

    results: list[CaseResult] = []
    rows, dim = 10, 1024
    x = make_fp16((rows, dim), 20260725, 0.2)
    style = make_fp16((rows, 3 * dim), 20260726, 0.1)
    packed, sfa = alloc_fp4(ops, rows, dim)
    packed.zero_()
    sfa.zero_()
    gate = torch.empty_like(x)
    ops.adaptive_rms_norm_nvfp4_fp16(x, style, packed, sfa, gate)
    torch.cuda.synchronize()

    scale, shift, gate_ref = style.chunk(3, dim=-1)
    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
    normed_ref = (
        x.float() * rstd * (1.0 + scale.float()) + shift.float()
    ).to(torch.float16)
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
        ops, packed, sfa, normed_ref
    )
    gate_equal = check_equal(gate, gate_ref)
    results.append(
        CaseResult(
            case="pi05_adaptive_rms_rows10_dim1024",
            rows=rows,
            dim=dim,
            check="adaptive_rms_norm_nvfp4_vs_math",
            packed_equal=True,
            sfa_equal=True,
            residual_equal=gate_equal,
            max_abs=max_abs,
            mean_abs=mean_abs,
            p99_abs=p99_abs,
            cosine=cosine,
            passed=(
                gate_equal
                and check_fp4_quant_reference_threshold(
                    max_abs, mean_abs, p99_abs, cosine
                )
            ),
        )
    )

    delta = make_fp16((rows, dim), 20260727, 0.1)
    previous_gate = make_fp16((rows, dim), 20260728, 0.1)
    residual = make_fp16((rows, dim), 20260729, 0.15)
    residual_ref = (
        residual.float() + delta.float() * previous_gate.float()
    ).to(torch.float16)
    packed.zero_()
    sfa.zero_()
    ops.gated_residual_adaptive_rms_norm_nvfp4_fp16(
        delta, previous_gate, residual, style, packed, sfa, gate
    )
    torch.cuda.synchronize()
    rstd = torch.rsqrt(
        residual_ref.float().square().mean(dim=-1, keepdim=True) + 1e-6
    )
    normed_ref = (
        residual_ref.float() * rstd * (1.0 + scale.float()) + shift.float()
    ).to(torch.float16)
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
        ops, packed, sfa, normed_ref
    )
    residual_equal = check_equal(residual, residual_ref)
    gate_equal = check_equal(gate, gate_ref)
    results.append(
        CaseResult(
            case="pi05_gated_residual_adaptive_rms_rows10_dim1024",
            rows=rows,
            dim=dim,
            check="gated_residual_adaptive_rms_nvfp4_vs_math",
            packed_equal=True,
            sfa_equal=True,
            residual_equal=residual_equal,
            max_abs=max_abs,
            mean_abs=mean_abs,
            p99_abs=p99_abs,
            cosine=cosine,
            passed=(
                residual_equal
                and gate_equal
                and check_fp4_quant_reference_threshold(
                    max_abs, mean_abs, p99_abs, cosine
                )
            ),
        )
    )

    rows, dim, eps = 768, 1152, 1e-5
    x = make_fp16((rows, dim), 20260730, 1.5)
    gamma = (make_fp16((1, dim), 20260731, 0.2).reshape(dim) + 1).contiguous()
    beta = make_fp16((1, dim), 20260732, 0.1).reshape(dim).contiguous()
    inv_s = (torch.rand(dim, device="cuda", dtype=torch.float16) + 0.5).contiguous()
    mean = x.float().mean(dim=-1, keepdim=True)
    var = (x.float() - mean).square().mean(dim=-1, keepdim=True)
    ln_ref = (
        (x.float() - mean) * torch.rsqrt(var + eps) * gamma.float()
        + beta.float()
    ).to(torch.float16)

    out_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ops.layer_norm_fp8_fp16(x, gamma, beta, eps, out_fp8)
    torch.cuda.synchronize()
    ref_fp8 = ln_ref.to(torch.float8_e4m3fn)
    fp8_diff = (out_fp8.float() - ref_fp8.float()).abs().flatten()
    fp8_cos = float(torch.nn.functional.cosine_similarity(
        out_fp8.float().flatten(), ref_fp8.float().flatten(), dim=0
    ))
    fp8_agree = float((out_fp8.view(torch.uint8) == ref_fp8.view(torch.uint8)).float().mean())
    results.append(
        CaseResult(
            case="siglip_layer_norm_rows768_dim1152",
            rows=rows,
            dim=dim,
            check="layer_norm_fp8_quantization_quality_vs_math",
            packed_equal=float(torch.quantile(fp8_diff, 0.99)) == 0.0,
            sfa_equal=True,
            residual_equal=None,
            max_abs=float(fp8_diff.max()),
            mean_abs=float(fp8_diff.mean()),
            p99_abs=float(torch.quantile(fp8_diff, 0.99)),
            cosine=fp8_cos,
            # Reduction order differs from eager LayerNorm at FP8 bin
            # boundaries.  The production acceptance test separately checks
            # >99.9% byte agreement against the native LayerNorm kernel.
            passed=(
                float(torch.quantile(fp8_diff, 0.99)) == 0.0
                and float(fp8_diff.mean()) <= 5e-4
                and fp8_cos >= 0.99998
            ),
        )
    )

    packed, sfa = alloc_fp4(ops, rows, dim)
    ops.layer_norm_nvfp4_fp16(x, gamma, beta, inv_s, eps, packed, sfa)
    awq_ref = (ln_ref.float() * inv_s.float()).to(torch.float16)
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
        ops, packed, sfa, awq_ref
    )
    results.append(
        CaseResult(
            case="siglip_layer_norm_awq_rows768_dim1152",
            rows=rows,
            dim=dim,
            check="layer_norm_nvfp4_vs_math",
            packed_equal=True,
            sfa_equal=True,
            residual_equal=None,
            max_abs=max_abs,
            mean_abs=mean_abs,
            p99_abs=p99_abs,
            cosine=cosine,
            passed=check_fp4_quant_reference_threshold(
                max_abs, mean_abs, p99_abs, cosine
            ),
        )
    )

    merged = make_fp16((51, 4096), 20260733, 0.4)
    packed, sfa = alloc_fp4(ops, 51, 2048)
    ops.gelu_mul_nvfp4_fp16(merged, packed, sfa)
    gate_values, up_values = merged.chunk(2, dim=-1)
    gelu_ref = (
        torch.nn.functional.gelu(
            gate_values.float(), approximate="tanh"
        )
        * up_values.float()
    ).to(torch.float16)
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
        ops, packed, sfa, gelu_ref
    )
    results.append(
        CaseResult(
            case="pi05_gelu_mul_rows51_dim2048",
            rows=51,
            dim=2048,
            check="gelu_mul_nvfp4_vs_math",
            packed_equal=True,
            sfa_equal=True,
            residual_equal=None,
            max_abs=max_abs,
            mean_abs=mean_abs,
            p99_abs=p99_abs,
            cosine=cosine,
            passed=check_fp4_quant_reference_threshold(
                max_abs, mean_abs, p99_abs, cosine
            ),
        )
    )
    return results


def run_pi05_thor_bf16_batch3_checks(ops) -> list[CaseResult]:
    """Gate every BF16 producer added for the PI0.5 Thor FP4 chain."""

    if torch.cuda.get_device_capability(0) != (11, 0):
        return []

    results: list[CaseResult] = []

    # B2: one-row style broadcast must be byte-identical to materializing the
    # same style for every decoder row. This also covers first-layer/no-residual
    # and recurrent gated-residual forms.
    rows, dim = 10, 1024
    x = make_fp16((rows, dim), 2026080801, 0.2).bfloat16()
    style_one = make_fp16((1, 3 * dim), 2026080802, 0.1).bfloat16()
    style_full = style_one.expand(rows, -1).contiguous()
    packed_one, sfa_one = alloc_fp4(ops, rows, dim)
    packed_full, sfa_full = alloc_fp4(ops, rows, dim)
    gate_one = torch.empty_like(x)
    gate_full = torch.empty_like(x)
    ops.adaptive_rms_norm_nvfp4_bf16(
        x, style_one, packed_one, sfa_one, gate_one
    )
    ops.adaptive_rms_norm_nvfp4_bf16(
        x, style_full, packed_full, sfa_full, gate_full
    )
    torch.cuda.synchronize()
    broadcast_exact = (
        torch.equal(packed_one, packed_full)
        and torch.equal(sfa_one, sfa_full)
        and torch.equal(gate_one, gate_full)
    )
    results.append(CaseResult(
        case="pi05_bf16_adarms_style_broadcast", rows=rows, dim=dim,
        check="one_row_style_equals_expanded_style",
        packed_equal=torch.equal(packed_one, packed_full),
        sfa_equal=torch.equal(sfa_one, sfa_full),
        residual_equal=torch.equal(gate_one, gate_full), max_abs=0.0,
        mean_abs=0.0, p99_abs=0.0, cosine=1.0,
        passed=broadcast_exact,
    ))

    delta = make_fp16((rows, dim), 2026080803, 0.1).bfloat16()
    previous_gate = make_fp16((rows, dim), 2026080804, 0.1).bfloat16()
    residual_seed = make_fp16((rows, dim), 2026080805, 0.15).bfloat16()
    residual_one = residual_seed.clone()
    residual_full = residual_seed.clone()
    packed_one.zero_()
    packed_full.zero_()
    sfa_one.zero_()
    sfa_full.zero_()
    ops.gated_residual_adaptive_rms_norm_nvfp4_bf16(
        delta, previous_gate, residual_one, style_one,
        packed_one, sfa_one, gate_one,
    )
    ops.gated_residual_adaptive_rms_norm_nvfp4_bf16(
        delta, previous_gate, residual_full, style_full,
        packed_full, sfa_full, gate_full,
    )
    torch.cuda.synchronize()
    gated_exact = (
        torch.equal(residual_one, residual_full)
        and torch.equal(packed_one, packed_full)
        and torch.equal(sfa_one, sfa_full)
        and torch.equal(gate_one, gate_full)
    )
    results.append(CaseResult(
        case="pi05_bf16_gate_res_adarms_style_broadcast", rows=rows,
        dim=dim, check="gated_one_row_style_equals_expanded_style",
        packed_equal=torch.equal(packed_one, packed_full),
        sfa_equal=torch.equal(sfa_one, sfa_full),
        residual_equal=torch.equal(residual_one, residual_full),
        max_abs=0.0, mean_abs=0.0, p99_abs=0.0, cosine=1.0,
        passed=gated_exact,
    ))

    # Capture only after eager warmup, matching production CUDA Graph setup.
    graph_residual = residual_seed.clone()
    graph_packed, graph_sfa = alloc_fp4(ops, rows, dim)
    graph_gate = torch.empty_like(x)
    ops.gated_residual_adaptive_rms_norm_nvfp4_bf16(
        delta, previous_gate, graph_residual, style_one,
        graph_packed, graph_sfa, graph_gate,
    )
    graph_residual.copy_(residual_seed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.gated_residual_adaptive_rms_norm_nvfp4_bf16(
            delta, previous_gate, graph_residual, style_one,
            graph_packed, graph_sfa, graph_gate,
        )
    graph_residual.copy_(residual_seed)
    graph.replay()
    torch.cuda.synchronize()
    first = (
        graph_residual.clone(), graph_packed.clone(), graph_sfa.clone(),
        graph_gate.clone(),
    )
    graph_residual.copy_(residual_seed)
    graph.replay()
    torch.cuda.synchronize()
    graph_exact = all(torch.equal(a, b) for a, b in zip(
        first, (graph_residual, graph_packed, graph_sfa, graph_gate)
    ))
    results.append(CaseResult(
        case="pi05_bf16_gate_res_adarms_graph", rows=rows, dim=dim,
        check="cuda_graph_replay_bit_exact", packed_equal=graph_exact,
        sfa_equal=graph_exact, residual_equal=graph_exact, max_abs=0.0,
        mean_abs=0.0, p99_abs=0.0, cosine=1.0, passed=graph_exact,
    ))

    # B3: BF16 GeGLU producer, including the optional AWQ inverse scale.
    rows, dim = 51, 2048
    merged = make_fp16((rows, 2 * dim), 2026080810, 0.3).bfloat16()
    inv_s = (make_fp16((1, dim), 2026080811, 0.05) + 1).reshape(dim).bfloat16()
    packed, sfa = alloc_fp4(ops, rows, dim)
    ops.gelu_mul_nvfp4_bf16(merged, inv_s, packed, sfa)
    gate_values, up_values = merged.chunk(2, dim=-1)
    ref = (
        torch.nn.functional.gelu(gate_values.float(), approximate="tanh")
        * up_values.float() * inv_s.float()
    ).bfloat16().half()
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
        ops, packed, sfa, ref
    )
    results.append(CaseResult(
        case="pi05_bf16_geglu_rows51_dim2048", rows=rows, dim=dim,
        check="bf16_geglu_nvfp4_vs_math", packed_equal=True, sfa_equal=True,
        residual_equal=None, max_abs=max_abs, mean_abs=mean_abs,
        p99_abs=p99_abs, cosine=cosine,
        passed=check_fp4_quant_reference_threshold(
            max_abs, mean_abs, p99_abs, cosine
        ),
    ))

    # B5: flat encoder RMS producers, with and without in-place residual.
    rows, dim, eps = 64, 2048, 1e-6
    x = make_fp16((rows, dim), 2026080820, 0.3).bfloat16()
    inv_s = (make_fp16((1, dim), 2026080821, 0.05) + 1).reshape(dim).bfloat16()
    packed, sfa = alloc_fp4(ops, rows, dim)
    ops.rms_norm_mul_nvfp4_bf16(x, inv_s, eps, packed, sfa)
    rms_ref = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        * inv_s.float()
    ).bfloat16().half()
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
        ops, packed, sfa, rms_ref
    )
    results.append(CaseResult(
        case="encoder_bf16_rms_mul_rows64_dim2048", rows=rows, dim=dim,
        check="bf16_rms_mul_nvfp4_vs_math", packed_equal=True,
        sfa_equal=True, residual_equal=None, max_abs=max_abs,
        mean_abs=mean_abs, p99_abs=p99_abs, cosine=cosine,
        passed=check_fp4_quant_reference_threshold(
            max_abs, mean_abs, p99_abs, cosine
        ),
    ))

    update = make_fp16((rows, dim), 2026080822, 0.1).bfloat16()
    residual_seed = make_fp16((rows, dim), 2026080823, 0.15).bfloat16()
    residual = residual_seed.clone()
    ops.residual_add_rms_norm_nvfp4_bf16(
        residual, update, inv_s, eps, packed, sfa
    )
    residual_ref = (residual_seed.float() + update.float()).bfloat16()
    rms_ref = (
        residual_ref.float()
        * torch.rsqrt(residual_ref.float().square().mean(-1, keepdim=True) + eps)
        * inv_s.float()
    ).bfloat16().half()
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
        ops, packed, sfa, rms_ref
    )
    residual_exact = torch.equal(residual, residual_ref)
    results.append(CaseResult(
        case="encoder_bf16_res_rms_mul_rows64_dim2048", rows=rows, dim=dim,
        check="residual_exact_and_bf16_rms_mul_nvfp4_vs_math",
        packed_equal=True, sfa_equal=True, residual_equal=residual_exact,
        max_abs=max_abs, mean_abs=mean_abs, p99_abs=p99_abs, cosine=cosine,
        passed=residual_exact and check_fp4_quant_reference_threshold(
            max_abs, mean_abs, p99_abs, cosine
        ),
    ))

    # B6: SigLIP BF16 LayerNorm producers at both production sequence bands.
    for rows in (512, 768):
        dim, eps = 1152, 1e-5
        x = make_fp16((rows, dim), 2026080830 + rows, 1.0).bfloat16()
        gamma = (make_fp16((1, dim), 2026080840 + rows, 0.1) + 1).reshape(dim).bfloat16()
        beta = make_fp16((1, dim), 2026080850 + rows, 0.1).reshape(dim).bfloat16()
        inv_s = (make_fp16((1, dim), 2026080860 + rows, 0.05) + 1).reshape(dim).bfloat16()
        mean = x.float().mean(-1, keepdim=True)
        var = (x.float() - mean).square().mean(-1, keepdim=True)
        ln_float_ref = ((x.float() - mean) * torch.rsqrt(var + eps)
                        * gamma.float() + beta.float())
        ln_ref = ln_float_ref.bfloat16()

        out_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        ops.layer_norm_fp8_bf16(x, gamma, beta, eps, out_fp8)
        # The fused FP8 producer quantizes the FP32 normalization result
        # directly, like the established FP16 vector kernel. There is no
        # materialized BF16 tensor at this seam.
        ref_fp8 = ln_float_ref.to(torch.float8_e4m3fn)
        diff = (out_fp8.float() - ref_fp8.float()).abs().flatten()
        cosine = float(torch.nn.functional.cosine_similarity(
            out_fp8.float().flatten(), ref_fp8.float().flatten(), dim=0
        ))
        p99_abs = float(torch.quantile(diff, 0.99))
        results.append(CaseResult(
            case=f"siglip_bf16_ln_fp8_rows{rows}_dim1152", rows=rows,
            dim=dim, check="bf16_layer_norm_fp8_vs_math",
            packed_equal=p99_abs == 0.0, sfa_equal=True,
            residual_equal=None, max_abs=float(diff.max()),
            mean_abs=float(diff.mean()), p99_abs=p99_abs, cosine=cosine,
            passed=p99_abs == 0.0 and float(diff.mean()) <= 5e-4
            and cosine >= 0.99998,
        ))

        packed, sfa = alloc_fp4(ops, rows, dim)
        ops.layer_norm_nvfp4_bf16(
            x, gamma, beta, inv_s, eps, packed, sfa
        )
        ref = (ln_ref.float() * inv_s.float()).bfloat16().half()
        max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
            ops, packed, sfa, ref
        )
        results.append(CaseResult(
            case=f"siglip_bf16_ln_fp4_rows{rows}_dim1152", rows=rows,
            dim=dim, check="bf16_layer_norm_nvfp4_vs_math",
            packed_equal=True, sfa_equal=True, residual_equal=None,
            max_abs=max_abs, mean_abs=mean_abs, p99_abs=p99_abs,
            cosine=cosine, passed=check_fp4_quant_reference_threshold(
                max_abs, mean_abs, p99_abs, cosine
            ),
        ))

    # B4: native LUT split-GU combiner must stay in the same quantized value
    # envelope as the established arithmetic path when inv_s is identity.
    rows, dim = 10, 4096
    raw_gate = make_fp16((rows, dim), 2026080870, 0.5)
    raw_up = make_fp16((rows, dim), 2026080871, 0.5)
    unit_gelu = float(torch.nn.functional.gelu(
        torch.tensor(1.0), approximate="tanh"
    ))
    # The package's existing GeGLU producer is used here only as an FP4 pack
    # oracle. A unit gate makes its output the requested raw distribution.
    gate_input = torch.cat((
        torch.ones_like(raw_gate), raw_gate / unit_gelu
    ), dim=-1).contiguous()
    up_input = torch.cat((
        torch.ones_like(raw_up), raw_up / unit_gelu
    ), dim=-1).contiguous()
    gate_packed, gate_sfa = alloc_fp4(ops, rows, dim)
    up_packed, up_sfa = alloc_fp4(ops, rows, dim)
    ops.silu_mul_fp4_sfa_v2_fp16(gate_input, gate_packed, gate_sfa)
    ops.silu_mul_fp4_sfa_v2_fp16(up_input, up_packed, up_sfa)
    native_packed, native_sfa = alloc_fp4(ops, rows, dim)
    inv_s = torch.ones(dim, device="cuda", dtype=torch.float16)
    ops.geglu_two_mul_nvfp4_native(
        gate_packed, gate_sfa, up_packed, up_sfa, inv_s,
        native_packed, native_sfa,
    )
    gate_deq = torch.empty((rows, dim), device="cuda", dtype=torch.float16)
    up_deq = torch.empty_like(gate_deq)
    ops.dequantize_fp4_sfa_fp16(gate_packed, gate_sfa, gate_deq)
    ops.dequantize_fp4_sfa_fp16(up_packed, up_sfa, up_deq)
    ref = (torch.nn.functional.gelu(
        gate_deq.float(), approximate="tanh"
    ) * up_deq.float()).half()
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(
        ops, native_packed, native_sfa, ref
    )
    results.append(CaseResult(
        case="pi05_split_gu_native_rows10_dim4096", rows=rows, dim=dim,
        check="native_lut_combiner_vs_dequant_math", packed_equal=True,
        sfa_equal=True, residual_equal=None, max_abs=max_abs,
        mean_abs=mean_abs, p99_abs=p99_abs, cosine=cosine,
        passed=check_fp4_quant_reference_threshold(
            max_abs, mean_abs, p99_abs, cosine
        ),
    ))
    return results


def encode_ue4m3_ceil(value: float) -> int:
    if value <= 0.0:
        return 0
    if value > 240.0:
        return 0xFE
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    float_exp = ((bits >> 23) & 0xFF) - 127
    fraction = bits & 0x7FFFFF
    ue_exp = float_exp + 7
    if ue_exp <= 0:
        mantissa = math.ceil(value * 512.0)
        if mantissa > 7:
            return 1 << 3
        return max(mantissa, 1)
    if ue_exp >= 15:
        return 0xFE
    mantissa = fraction >> 20
    if fraction & 0xFFFFF:
        mantissa += 1
    if mantissa >= 8:
        mantissa = 0
        ue_exp += 1
    if ue_exp >= 15:
        return 0xFE
    return (ue_exp << 3) | mantissa


def decode_ue4m3(value: int) -> float:
    exponent = (value >> 3) & 0xF
    mantissa = value & 0x7
    if exponent == 0:
        return math.ldexp(mantissa / 8.0, -6)
    return math.ldexp(1.0 + mantissa / 8.0, exponent - 7)


def encode_e2m1(value: float) -> int:
    magnitude = abs(value)
    thresholds = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    encoded = sum(magnitude >= threshold for threshold in thresholds)
    return (0x8 if value < 0.0 else 0) | encoded


def linear_nvfp4_reference(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cpu = input.float().cpu()
    rows, cols = cpu.shape
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8)
    scales = torch.empty((rows, cols // 16), dtype=torch.uint8)
    for row in range(rows):
        for block in range(cols // 16):
            values = cpu[row, block * 16 : (block + 1) * 16]
            scale_byte = encode_ue4m3_ceil(float(values.abs().max()) / 6.0)
            scales[row, block] = scale_byte
            scale = decode_ue4m3(scale_byte)
            inverse = 1.0 / scale if scale > 0.0 else 0.0
            for pair in range(8):
                index = block * 16 + pair * 2
                low = encode_e2m1(float(cpu[row, index]) * inverse)
                high = encode_e2m1(float(cpu[row, index + 1]) * inverse)
                packed[row, index // 2] = (high << 4) | low
    return packed.to(input.device), scales.to(input.device)


def dequantize_linear_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.device,
    )
    low = packed & 0xF
    high = packed >> 4
    low_value = magnitude[(low & 0x7).long()] * torch.where(
        low & 0x8 != 0, -1.0, 1.0
    )
    high_value = magnitude[(high & 0x7).long()] * torch.where(
        high & 0x8 != 0, -1.0, 1.0
    )
    values = torch.stack((low_value, high_value), dim=-1).flatten(-2)
    exponent = (scales >> 3) & 0xF
    mantissa = scales & 0x7
    normal = torch.ldexp(
        1.0 + mantissa.float() / 8.0, exponent.to(torch.int32) - 7
    )
    subnormal = torch.ldexp(
        mantissa.float() / 8.0,
        torch.full_like(exponent, -6, dtype=torch.int32),
    )
    decoded_scales = torch.where(exponent == 0, subnormal, normal)
    return values * decoded_scales.repeat_interleave(16, dim=-1)


def run_linear_nvfp4_checks(ops: SourceOps) -> list[CaseResult]:
    results = []
    for rows, cols in ((1, 128), (3, 320), (17, 1024)):
        input = make_fp16((rows, cols), 12000 + rows + cols).to(
            torch.bfloat16
        )
        got_packed, got_scales = ops.quantize_bf16_to_nvfp4_linear(input)
        ref_packed, ref_scales = linear_nvfp4_reference(input)
        packed_equal = check_equal(got_packed, ref_packed)
        scale_equal = check_equal(got_scales, ref_scales)
        dequant = dequantize_linear_nvfp4(got_packed, got_scales)
        diff = (dequant - input.float()).abs().flatten()
        cosine = float(
            torch.nn.functional.cosine_similarity(
                dequant.flatten(), input.float().flatten(), dim=0
            )
        )
        results.append(
            CaseResult(
                case=f"linear_nvfp4_rows{rows}_cols{cols}",
                rows=rows,
                dim=cols,
                check="bit_exact_reference_and_dequant",
                packed_equal=packed_equal,
                sfa_equal=scale_equal,
                residual_equal=None,
                max_abs=float(diff.max()),
                mean_abs=float(diff.mean()),
                p99_abs=float(torch.quantile(diff, 0.99)),
                cosine=cosine,
                passed=packed_equal and scale_equal and cosine >= 0.99,
            )
        )

    x = make_fp16((1, 128 * 2 * 2), 13000).to(torch.bfloat16).reshape(
        1, 128, 2, 2, 1
    )
    gamma = make_fp16((1, 128), 13001).reshape(128).to(torch.bfloat16)
    packed, scales = ops.rms_silu_nvfp4_ndhwc_bf16(x, gamma)
    inv_rms = torch.rsqrt(x.float().square().mean(dim=1, keepdim=True) + 1e-6)
    normalized = (x.float() * inv_rms * gamma.float().view(1, -1, 1, 1, 1)).to(
        torch.bfloat16
    )
    reference = torch.nn.functional.silu(normalized.float()).to(
        torch.bfloat16
    )
    reference_rows = reference.permute(0, 2, 3, 4, 1).reshape(-1, 128)
    ref_packed, ref_scales = linear_nvfp4_reference(reference_rows)
    producer_equal = check_equal(packed.reshape(-1, 64), ref_packed)
    producer_scales_equal = check_equal(scales.reshape(-1, 8), ref_scales)
    dequant = dequantize_linear_nvfp4(
        packed.reshape(-1, 64), scales.reshape(-1, 8)
    )
    diff = (dequant - reference_rows.float()).abs().flatten()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            dequant.flatten(), reference_rows.float().flatten(), dim=0
        )
    )
    results.append(
        CaseResult(
            case="rms_silu_nvfp4_b1_c128_t2_h2_w1",
            rows=4,
            dim=128,
            check="fused_producer_bit_exact_reference",
            packed_equal=producer_equal,
            sfa_equal=producer_scales_equal,
            residual_equal=None,
            max_abs=float(diff.max()),
            mean_abs=float(diff.mean()),
            p99_abs=float(torch.quantile(diff, 0.99)),
            cosine=cosine,
            passed=producer_equal and producer_scales_equal and cosine >= 0.99,
        )
    )
    return results


def run_case(ops: SourceOps, name: str, rows: int, dim: int) -> list[CaseResult]:
    results: list[CaseResult] = []

    residual = make_fp16((rows, dim), seed=1000 + rows + dim)
    x = make_fp16((rows, dim), seed=2000 + rows + dim)
    residual_v1 = residual.clone()
    residual_v2 = residual.clone()
    packed_v1, sfa_v1 = alloc_fp4(ops, rows, dim)
    packed_v2, sfa_v2 = alloc_fp4(ops, rows, dim)
    if dim <= 2048:
        ops.residual_add_rms_norm_fp4_sfa_fp16(residual_v1, x, packed_v1, sfa_v1)
    ops.residual_add_rms_norm_fp4_sfa_v2_fp16(residual_v2, x, packed_v2, sfa_v2)
    torch.cuda.synchronize()
    residual_ref = (residual.float() + x.float()).to(torch.float16)
    residual_equal = check_equal(residual_v2, residual_ref)
    norm_ref = ((residual.float() + x.float()) * torch.rsqrt((residual.float() + x.float()).pow(2).mean(dim=1, keepdim=True) + 1e-6)).to(torch.float16)
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics_vs_ref(ops, packed_v2, sfa_v2, norm_ref)
    dequant_passed = check_fp4_quant_reference_threshold(max_abs, mean_abs, p99_abs, cosine)
    packed_equal = False
    sfa_equal = False
    if dim <= 2048:
        packed_equal = check_equal(packed_v1, packed_v2)
        sfa_equal = check_equal(sfa_v1, sfa_v2)
    results.append(
        CaseResult(
            case=name,
            rows=rows,
            dim=dim,
            check="residual_add_rms_norm_v2_vs_math_reference",
            packed_equal=packed_equal,
            sfa_equal=sfa_equal,
            residual_equal=residual_equal,
            max_abs=max_abs,
            mean_abs=mean_abs,
            p99_abs=p99_abs,
            cosine=cosine,
            passed=dequant_passed and residual_equal,
        )
    )

    merged = make_fp16((rows, dim * 2), seed=3000 + rows + dim)
    packed_v1, sfa_v1 = alloc_fp4(ops, rows, dim)
    packed_v2, sfa_v2 = alloc_fp4(ops, rows, dim)
    ops.silu_mul_fp4_sfa_fp16(merged, packed_v1, sfa_v1)
    ops.silu_mul_fp4_sfa_v2_fp16(merged, packed_v2, sfa_v2)
    torch.cuda.synchronize()
    packed_equal = check_equal(packed_v1, packed_v2)
    sfa_equal = check_equal(sfa_v1, sfa_v2)
    max_abs, mean_abs, p99_abs, cosine = dequant_metrics(ops, packed_v1, sfa_v1, packed_v2, sfa_v2)
    path_passed = check_fp4_path_equivalence_threshold(max_abs, mean_abs, p99_abs, cosine)
    gate = merged[:, :dim].float()
    up = merged[:, dim:].float()
    gelu = gate / (1.0 + torch.exp(-1.5957691216057308 * gate * (1.0 + 0.044715 * gate * gate)))
    f4_ref = (gelu * up).to(torch.float16)
    ref_max, ref_mean, ref_p99, ref_cos = dequant_metrics_vs_ref(ops, packed_v2, sfa_v2, f4_ref)
    ref_passed = ref_max <= 0.1 and ref_mean <= 0.004 and ref_p99 <= 0.02 and ref_cos >= 0.99
    results.append(
        CaseResult(
            case=name,
            rows=rows,
            dim=dim,
            check="silu_mul_v2_vs_v1_dequant",
            packed_equal=packed_equal,
            sfa_equal=sfa_equal,
            residual_equal=None,
            max_abs=max_abs,
            mean_abs=mean_abs,
            p99_abs=p99_abs,
            cosine=cosine,
            passed=path_passed and ref_passed,
        )
    )

    inv_s = (torch.rand((dim,), device="cuda") * 0.25 + 0.875).to(torch.float16).contiguous()
    residual_awq = residual.clone()
    packed_awq, sfa_awq = alloc_fp4(ops, rows, dim)
    if dim <= 2048:
        ops.residual_add_rms_norm_mul_fp4_sfa_fp16(residual_awq, x, inv_s, packed_awq, sfa_awq)
        torch.cuda.synchronize()
        residual_mul_passed = bool(
            int(packed_awq.sum().item()) != 0
            and int(sfa_awq.sum().item()) != 0
            and torch.equal(residual_awq, residual_ref)
        )
        residual_mul_rejected = False
    else:
        try:
            ops.residual_add_rms_norm_mul_fp4_sfa_fp16(residual_awq, x, inv_s, packed_awq, sfa_awq)
            residual_mul_passed = False
            residual_mul_rejected = False
        except RuntimeError as exc:
            residual_mul_passed = "dim <= 2048" in str(exc)
            residual_mul_rejected = True
    results.append(
        CaseResult(
            case=name,
            rows=rows,
            dim=dim,
            check="residual_add_rms_norm_mul_smoke_or_reject",
            packed_equal=bool(residual_mul_passed),
            sfa_equal=bool(residual_mul_passed),
            residual_equal=bool(residual_mul_passed or residual_mul_rejected),
            max_abs=None,
            mean_abs=None,
            p99_abs=None,
            cosine=None,
            passed=residual_mul_passed,
        )
    )

    packed_awq, sfa_awq = alloc_fp4(ops, rows, dim)
    ops.silu_mul_mul_fp4_sfa_v2_fp16(merged, inv_s, packed_awq, sfa_awq)
    torch.cuda.synchronize()
    results.append(
        CaseResult(
            case=name,
            rows=rows,
            dim=dim,
            check="silu_mul_mul_v2_smoke_nonzero",
            packed_equal=bool(packed_awq.numel() == rows * dim // 2 and int(packed_awq.sum().item()) != 0),
            sfa_equal=bool(sfa_awq.numel() >= ops.sfa_size_bytes(rows, dim, False) and int(sfa_awq.sum().item()) != 0),
            residual_equal=None,
            max_abs=None,
            mean_abs=None,
            p99_abs=None,
            cosine=None,
            passed=bool(int(packed_awq.sum().item()) != 0 and int(sfa_awq.sum().item()) != 0),
        )
    )

    gate_packed, gate_sfa = alloc_fp4(ops, rows, dim)
    up_packed, up_sfa = alloc_fp4(ops, rows, dim)
    out_packed, out_sfa = alloc_fp4(ops, rows, dim)
    ops.silu_mul_fp4_sfa_v2_fp16(make_fp16((rows, dim * 2), seed=4000 + rows + dim), gate_packed, gate_sfa)
    ops.silu_mul_fp4_sfa_v2_fp16(make_fp16((rows, dim * 2), seed=5000 + rows + dim), up_packed, up_sfa)
    ops.silu_mul_two_fp4_to_fp4(gate_packed, gate_sfa, up_packed, up_sfa, out_packed, out_sfa)
    torch.cuda.synchronize()
    results.append(
        CaseResult(
            case=name,
            rows=rows,
            dim=dim,
            check="silu_mul_two_fp4_to_fp4_smoke_nonzero",
            packed_equal=bool(int(out_packed.sum().item()) != 0),
            sfa_equal=bool(int(out_sfa.sum().item()) != 0),
            residual_equal=None,
            max_abs=None,
            mean_abs=None,
            p99_abs=None,
            cosine=None,
            passed=bool(int(out_packed.sum().item()) != 0 and int(out_sfa.sum().item()) != 0),
        )
    )

    out_mul_packed, out_mul_sfa = alloc_fp4(ops, rows, dim)
    ops.silu_mul_two_mul_fp4_to_fp4(gate_packed, gate_sfa, up_packed, up_sfa, inv_s, out_mul_packed, out_mul_sfa)
    torch.cuda.synchronize()
    results.append(
        CaseResult(
            case=name,
            rows=rows,
            dim=dim,
            check="silu_mul_two_mul_fp4_to_fp4_smoke_nonzero",
            packed_equal=bool(int(out_mul_packed.sum().item()) != 0),
            sfa_equal=bool(int(out_mul_sfa.sum().item()) != 0),
            residual_equal=None,
            max_abs=None,
            mean_abs=None,
            p99_abs=None,
            cosine=None,
            passed=bool(int(out_mul_packed.sum().item()) != 0 and int(out_mul_sfa.sum().item()) != 0),
        )
    )

    return results


def run_unsupported_checks(ops: SourceOps) -> list[CaseResult]:
    results = []
    x = make_fp16((1, 1025), seed=9000)
    try:
        packed = torch.empty((1, 512), device="cuda", dtype=torch.uint8)
        sfa = torch.empty((1024,), device="cuda", dtype=torch.uint8)
        ops.rms_norm_fp4_sfa_fp16(x, packed, sfa)
        passed = False
    except RuntimeError as exc:
        passed = "divisible by 16" in str(exc)
    results.append(
        CaseResult(
            case="unsupported_dim1025",
            rows=1,
            dim=1025,
            check="unsupported_shape_rejected",
            packed_equal=passed,
            sfa_equal=passed,
            residual_equal=None,
            max_abs=None,
            mean_abs=None,
            p99_abs=None,
            cosine=None,
            passed=passed,
        )
    )
    x2 = make_fp16((1, 4096), seed=9001)
    packed2 = torch.empty((1, 2048), device="cuda", dtype=torch.uint8)
    sfa2 = torch.empty((ops.sfa_size_bytes(1, 4096),), device="cuda", dtype=torch.uint8)
    try:
        ops.rms_norm_fp4_sfa_fp16(x2, packed2, sfa2)
        passed = False
    except RuntimeError as exc:
        passed = "dim <= 2048" in str(exc)
    results.append(
        CaseResult(
            case="unsupported_v1_dim4096",
            rows=1,
            dim=4096,
            check="v1_dim_limit_rejected",
            packed_equal=passed,
            sfa_equal=passed,
            residual_equal=None,
            max_abs=None,
            mean_abs=None,
            p99_abs=None,
            cosine=None,
            passed=passed,
        )
    )
    return results


def _distribution_metrics(
    got: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float, float, float]:
    diff = (got.float() - expected.float()).abs().flatten()
    sorted_diff = diff.sort().values
    p99_index = min(
        sorted_diff.numel() - 1,
        max(0, math.ceil(0.99 * sorted_diff.numel()) - 1),
    )
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(sorted_diff[p99_index].item()),
        float(
            torch.nn.functional.cosine_similarity(
                got.float().flatten(),
                expected.float().flatten(),
                dim=0,
            ).item()
        ),
    )


def _ref_rms_ncdhw(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(dim=1, keepdim=True) + eps)
    result = xf * inv_rms * gamma.float().view(1, -1, 1, 1, 1)
    if bias is not None:
        result = result + bias.float().view(1, -1, 1, 1, 1)
    return result.to(torch.bfloat16)


def run_ncdhw_bf16_checks(ops) -> list[CaseResult]:
    results: list[CaseResult] = []
    shapes = [
        ("cosmos_t1_c64", (1, 64, 1, 17, 19)),
        ("cosmos_t5_c128", (1, 128, 5, 9, 11)),
        ("motus_c320", (1, 320, 3, 7, 8)),
        ("world_c512", (1, 512, 1, 5, 6)),
        ("boundary_c1024", (1, 1024, 1, 3, 4)),
    ]
    eps = 1e-6
    for index, (name, shape) in enumerate(shapes):
        generator = torch.Generator(device="cuda").manual_seed(12000 + index)
        x = torch.randn(
            shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        gamma = torch.randn(
            (shape[1],),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        bias = torch.randn(
            (shape[1],),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        got_norm = ops.bf16_rms_norm_ncdhw(x, gamma, bias, eps)
        ref_norm = _ref_rms_ncdhw(x, gamma, eps, bias)
        max_abs, mean_abs, p99_abs, cosine = _distribution_metrics(
            got_norm, ref_norm
        )
        norm_passed = p99_abs == 0.0 and cosine >= 0.99999
        results.append(
            CaseResult(
                case=name,
                rows=shape[0] * shape[2] * shape[3] * shape[4],
                dim=shape[1],
                check="bf16_rms_norm_ncdhw",
                packed_equal=norm_passed,
                sfa_equal=norm_passed,
                residual_equal=None,
                max_abs=max_abs,
                mean_abs=mean_abs,
                p99_abs=p99_abs,
                cosine=cosine,
                passed=norm_passed,
            )
        )

        prev_cache = torch.randn(
            (shape[0], shape[1], 2, shape[3], shape[4]),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        next_cache = torch.empty_like(prev_cache)
        got_silu, got_cache = ops.bf16_rms_silu_ncdhw(
            x,
            gamma,
            prev_cache,
            eps,
            next_cache=next_cache,
        )
        norm_rounded = _ref_rms_ncdhw(x, gamma, eps)
        ref_silu = torch.nn.functional.silu(norm_rounded.float()).to(
            torch.bfloat16
        )
        max_abs, mean_abs, p99_abs, cosine = _distribution_metrics(
            got_silu, ref_silu
        )
        if shape[2] == 1:
            expected_cache = torch.stack(
                (prev_cache[:, :, 1], ref_silu[:, :, 0]), dim=2
            )
        else:
            expected_cache = ref_silu[:, :, -2:]
        cache_equal = torch.equal(got_cache, expected_cache)
        silu_passed = p99_abs == 0.0 and cosine >= 0.99999
        results.append(
            CaseResult(
                case=name,
                rows=shape[0] * shape[2] * shape[3] * shape[4],
                dim=shape[1],
                check="bf16_rms_silu_ncdhw_cache",
                packed_equal=silu_passed,
                sfa_equal=cache_equal,
                residual_equal=cache_equal,
                max_abs=max_abs,
                mean_abs=mean_abs,
                p99_abs=p99_abs,
                cosine=cosine,
                passed=silu_passed and cache_equal,
            )
        )

        compiled = torch.compile(
            lambda value, weight, offset: ops.bf16_rms_norm_ncdhw(
                value, weight, offset, eps
            ),
            fullgraph=True,
        )
        compiled_out = compiled(x, gamma, bias)
        compile_equal = torch.equal(compiled_out, got_norm)
        results.append(
            CaseResult(
                case=name,
                rows=shape[0] * shape[2] * shape[3] * shape[4],
                dim=shape[1],
                check="bf16_rms_norm_ncdhw_compile",
                packed_equal=compile_equal,
                sfa_equal=compile_equal,
                residual_equal=None,
                max_abs=0.0 if compile_equal else None,
                mean_abs=0.0 if compile_equal else None,
                p99_abs=0.0 if compile_equal else None,
                cosine=1.0 if compile_equal else None,
                passed=compile_equal,
            )
        )

    bad_x = torch.empty(
        (1, 65, 1, 2, 2), device="cuda", dtype=torch.bfloat16
    )
    bad_gamma = torch.empty((65,), device="cuda", dtype=torch.bfloat16)
    try:
        ops.bf16_rms_norm_ncdhw(bad_x, bad_gamma)
        rejected = False
    except RuntimeError as exc:
        rejected = "C must be even" in str(exc)
    results.append(
        CaseResult(
            case="unsupported_ncdhw_c65",
            rows=4,
            dim=65,
            check="unsupported_ncdhw_rejected",
            packed_equal=rejected,
            sfa_equal=rejected,
            residual_equal=None,
            max_abs=None,
            mean_abs=None,
            p99_abs=None,
            cosine=None,
            passed=rejected,
        )
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    results: list[CaseResult] = []
    selected_shapes = THOR_MODEL_SHAPES if args.mode == "thor-models" else SHAPES
    for name in MODES[args.mode]:
        rows, dim = selected_shapes[name]
        results.extend(run_case(ops, name, rows, dim))
    results.extend(run_linear_nvfp4_checks(ops))
    results.extend(run_ncdhw_bf16_checks(ops))
    results.extend(run_fp8_adarms_checks(ops))
    results.extend(run_bf16_adarms_nvfp4_checks(ops))
    results.extend(run_e0m3_and_cosmos_fp4_checks(ops))
    results.extend(run_unsupported_checks(ops))
    results.extend(run_pi05_thor_producer_checks(ops))
    results.extend(run_pi05_thor_bf16_batch3_checks(ops))
    if args.mode == "full":
        run_request2_checks(ops)

    passed = sum(1 for item in results if item.passed)
    payload = {
        "backend": args.backend,
        "mode": args.mode,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "passed": passed,
        "total": len(results),
        "results": [asdict(item) for item in results],
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
