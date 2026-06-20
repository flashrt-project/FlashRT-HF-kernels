#!/usr/bin/env python3
"""Correctness tests for fp4-fused-ops."""

from __future__ import annotations

import argparse
import importlib
import json
import os
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
            str(PACKAGE / "csrc" / "quantize" / "reshape_scales_sfa.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(PACKAGE / "csrc" / "quantize"), str(cutlass_include), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DCUDA_KERNEL",
            "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED",
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


def run_case(ops: SourceOps, name: str, rows: int, dim: int) -> list[CaseResult]:
    results: list[CaseResult] = []

    residual = make_fp16((rows, dim), seed=1000 + rows + dim)
    x = make_fp16((rows, dim), seed=2000 + rows + dim)
    residual_v1 = residual.clone()
    residual_v2 = residual.clone()
    packed_v1, sfa_v1 = ops.alloc(rows, dim)
    packed_v2, sfa_v2 = ops.alloc(rows, dim)
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
    packed_v1, sfa_v1 = ops.alloc(rows, dim)
    packed_v2, sfa_v2 = ops.alloc(rows, dim)
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
    packed_awq, sfa_awq = ops.alloc(rows, dim)
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

    packed_awq, sfa_awq = ops.alloc(rows, dim)
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

    gate_packed, gate_sfa = ops.alloc(rows, dim)
    up_packed, up_sfa = ops.alloc(rows, dim)
    out_packed, out_sfa = ops.alloc(rows, dim)
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

    out_mul_packed, out_mul_sfa = ops.alloc(rows, dim)
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
