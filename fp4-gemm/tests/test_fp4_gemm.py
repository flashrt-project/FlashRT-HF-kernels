#!/usr/bin/env python3
"""Correctness tests for fp4-gemm."""

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
PACKAGE = ROOT / "fp4-gemm"
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
DEFAULT_CUTLASS_UTIL_INCLUDE = (
    ROOT.parent
    / "flashrt_pr31_review"
    / "third_party"
    / "cutlass"
    / "tools"
    / "util"
    / "include"
)


SHAPES = {
    "small_m16_n128_k128": (16, 128, 128),
    "small_m32_n256_k256": (32, 256, 256),
    "mlp_tile_m64_n512_k512": (64, 512, 512),
}

MODES = {
    "smoke": ["small_m16_n128_k128"],
    "full": list(SHAPES),
}


@dataclass
class Metrics:
    shape: str
    M: int
    N: int
    K: int
    workload: str
    variant: int | None
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    passed: bool


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    @staticmethod
    def sfa_size_bytes(rows: int, dim: int) -> int:
        n_blocks = dim // 16
        n_row_super = (rows + 127) // 128
        n_col_super = (n_blocks + 3) // 4
        return n_row_super * n_col_super * 512

    def alloc_fp4(self, rows: int, dim: int):
        return (
            torch.empty((rows, dim // 2), device="cuda", dtype=torch.uint8),
            torch.empty((self.sfa_size_bytes(rows, dim),), device="cuda", dtype=torch.uint8),
        )

    def quantize_fp4_sfa_fp16(self, x, packed, sfa, is_sfb=False):
        self._ops.quantize_fp4_sfa_fp16(x, packed, sfa, bool(is_sfb))

    def dequantize_fp4_sfa_fp16(self, packed, sfa, out, is_sfb=False):
        self._ops.dequantize_fp4_sfa_fp16(packed, sfa, out, bool(is_sfb))

    def fp4_w4a16_linear_bf16(self, a, b, sfa, sfb, out, alpha=1.0, variant=0):
        self._ops.fp4_w4a16_linear_bf16(a, b, sfa, sfb, out, float(alpha), int(variant))


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 12:
        return "12.0a"
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    cutlass_include = Path(os.environ.get("FLASHRT_CUTLASS_INCLUDE", str(DEFAULT_CUTLASS_INCLUDE)))
    cutlass_util_include = Path(os.environ.get("FLASHRT_CUTLASS_UTIL_INCLUDE", str(DEFAULT_CUTLASS_UTIL_INCLUDE)))
    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    if not cutlass_include.is_dir():
        raise RuntimeError(f"missing CUTLASS include path: {cutlass_include}")
    if not cutlass_util_include.is_dir():
        raise RuntimeError(f"missing CUTLASS util include path: {cutlass_util_include}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "fp4_gemm_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "gemm" / "fp4" / "cutlass_nvfp4_w4a16_gemm_sm120.cu"),
            str(PACKAGE / "csrc" / "quantize" / "quantize_fp4_sfa.cu"),
            str(PACKAGE / "csrc" / "dequantize_fp4_sfa.cu"),
        ],
        extra_include_paths=[
            str(PACKAGE / "csrc"),
            str(cutlass_include),
            str(cutlass_util_include),
            str(REGISTRATION_INCLUDE),
        ],
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
        return importlib.import_module("fp4_gemm")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_inputs(m: int, n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    a = (torch.randn((m, k), device="cuda", generator=gen) * 0.25).to(torch.float16).contiguous()
    b = (torch.randn((n, k), device="cuda", generator=gen) * 0.25).to(torch.float16).contiguous()
    return a, b


def metrics(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (got.float() - expected.float()).abs().flatten()
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(torch.quantile(diff, 0.99).item()),
        float(torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item()),
    )


def check_bf16_threshold(max_abs: float, mean_abs: float, p99_abs: float, cosine: float) -> bool:
    return max_abs <= 0.125 and mean_abs <= 0.005 and p99_abs <= 0.03125 and cosine >= 0.999


def prepare_quantized(ops: SourceOps, m: int, n: int, k: int):
    a_fp16, b_fp16 = make_inputs(m, n, k, seed=7000 + m + n + k)
    a_packed, sfa = ops.alloc_fp4(m, k)
    b_packed, sfb = ops.alloc_fp4(n, k)
    ops.quantize_fp4_sfa_fp16(a_fp16, a_packed, sfa, False)
    ops.quantize_fp4_sfa_fp16(b_fp16, b_packed, sfb, True)
    a_deq = torch.empty_like(a_fp16)
    b_deq = torch.empty_like(b_fp16)
    ops.dequantize_fp4_sfa_fp16(a_packed, sfa, a_deq, False)
    ops.dequantize_fp4_sfa_fp16(b_packed, sfb, b_deq, True)
    torch.cuda.synchronize()
    expected = (a_deq.float() @ b_deq.float().T).to(torch.bfloat16)
    return a_packed, b_packed, sfa, sfb, expected


def run_case(ops: SourceOps, name: str, shape: tuple[int, int, int]) -> list[Metrics]:
    m, n, k = shape
    a_packed, b_packed, sfa, sfb, expected = prepare_quantized(ops, m, n, k)
    results: list[Metrics] = []
    for variant in (0, 1, 2):
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        ops.fp4_w4a16_linear_bf16(a_packed, b_packed, sfa, sfb, out, 1.0, variant)
        torch.cuda.synchronize()
        max_abs, mean_abs, p99_abs, cosine = metrics(out, expected)
        results.append(
            Metrics(
                shape=name,
                M=m,
                N=n,
                K=k,
                workload="fp4_w4a16_linear_bf16",
                variant=variant,
                max_abs=max_abs,
                mean_abs=mean_abs,
                p99_abs=p99_abs,
                cosine=cosine,
                passed=check_bf16_threshold(max_abs, mean_abs, p99_abs, cosine),
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

    results: list[Metrics] = []
    for name in MODES[args.mode]:
        results.extend(run_case(ops, name, SHAPES[name]))
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
