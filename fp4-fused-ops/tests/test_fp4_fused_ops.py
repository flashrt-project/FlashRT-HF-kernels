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

MODES = {
    "smoke": ["tiny_rows1_dim1024", "decode_rows10_dim2048"],
    "full": list(SHAPES),
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
            torch.empty((self.sfa_size_bytes(rows, dim, False),), device=device, dtype=torch.uint8),
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
        torch.empty(
            (ops.sfa_size_bytes(rows, dim, False),),
            device="cuda",
            dtype=torch.uint8,
        ),
    )


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
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
            str(PACKAGE / "csrc" / "fused_fp4" / "norm_silu_fp4_sfa.cu"),
            str(PACKAGE / "csrc" / "fused_fp4" / "dequantize_fp4_sfa.cu"),
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
        extra_include_paths=[str(PACKAGE / "csrc"), str(PACKAGE / "csrc" / "quantize"), str(cutlass_include), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DCUDA_KERNEL",
        ],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("fp4_fused_ops")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_fp16(shape: tuple[int, int], seed: int, scale: float = 0.25) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=gen) * scale).to(torch.float16).contiguous()


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
    for name in MODES[args.mode]:
        rows, dim = SHAPES[name]
        results.extend(run_case(ops, name, rows, dim))
    results.extend(run_linear_nvfp4_checks(ops))
    results.extend(run_ncdhw_bf16_checks(ops))
    results.extend(run_unsupported_checks(ops))

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
