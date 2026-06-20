#!/usr/bin/env python3
"""Correctness tests for fp8-gemm."""

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


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 12:
        return "12.0a"
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "fp8_gemm_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "fp8_gemv_m1_sm120.cu"),
            str(PACKAGE / "csrc" / "fp8_smallM_handtuned_sm120.cu"),
            str(PACKAGE / "csrc" / "fp8_smallM_handtuned_ldmatrix_sm120.cu"),
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
        return importlib.import_module("fp8_gemm")
    finally:
        if artifact:
            sys.path.remove(artifact)


def select_tile(m: int, n: int, k: int, variant: int = 0) -> str:
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
    raise RuntimeError("unsupported M")


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
    p99_abs = float(torch.quantile(diff, 0.99).item())
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
    ops.fp8_linear_residual_bf16(x, w, got, alpha=1.0, variant=8)
    torch.cuda.synchronize()
    max_abs, mean_abs, p99_abs, cos = compare(got, expected)
    passed = check_threshold(max_abs, mean_abs, p99_abs, cos)
    return Metrics(
        shape="decode_residual_m1_k4096_n2048",
        M=m,
        K=k,
        N=n,
        variant=8,
        tile="gemv_fp8_m1_resadd_w8",
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine=cos,
        dtype=str(got.dtype),
        tolerance="max_abs<=0.5 mean_abs<=0.02 p99_abs<=0.25 cosine>=0.999",
        passed=passed,
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
    major, _minor = torch.cuda.get_device_capability(0)
    if major < 12:
        raise SystemExit("fp8-gemm requires Blackwell/SM120 for this package")

    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    rows = [run_case(ops, name, SHAPES[name]) for name in MODES[args.mode]]
    rows.append(run_residual_case(ops))

    failed = [row for row in rows if not row.passed]
    payload = {"passed": len(rows) - len(failed), "failed": len(failed), "rows": [asdict(row) for row in rows]}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
