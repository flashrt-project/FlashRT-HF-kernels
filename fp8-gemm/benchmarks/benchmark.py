#!/usr/bin/env python3
"""Benchmark fp8-gemm."""

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
    "decode_m1_k4096_n2048": (1, 4096, 2048),
    "decode_m1_k4096_n8192": (1, 4096, 8192),
    "small_m16_k4096_n4096": (16, 4096, 4096),
    "small_m32_k4096_n8192": (32, 4096, 8192),
    "small_m64_k512_n1024": (64, 512, 1024),
}

MODES = {
    "smoke": ["decode_m1_k4096_n2048", "small_m16_k4096_n4096"],
    "headline": list(SHAPES),
}


@dataclass
class Result:
    shape: str
    M: int
    K: int
    N: int
    variant: int
    tile: str
    flashrt_us: float
    torch_eager_us: float
    torch_compile_us: float | None
    speedup_vs_eager: float
    speedup_vs_compile: float | None
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    status: str


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


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 12:
        return "12.0a"
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "fp8_gemm_source_bench"
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
    x = (torch.randn((m, k), device="cuda", generator=gen) * 0.25).to(torch.bfloat16).to(torch.float8_e4m3fn)
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.25).to(torch.bfloat16).to(torch.float8_e4m3fn)
    return x, w


def ref_fn(x, w):
    return (x.float() @ w.float().T).to(torch.bfloat16)


def measure(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iters)


def metrics(got, expected):
    diff = (got.float() - expected.float()).abs().flatten()
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(torch.quantile(diff, 0.99).item()),
        float(torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item()),
    )


def bench_case(ops, name: str, shape: tuple[int, int, int], variant: int, warmup: int, iters: int, compile_ref: bool):
    m, k, n = shape
    x, w = make_inputs(m, k, n, seed=3000 + m + k + n + variant)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    expected = ref_fn(x, w)
    got = ops.fp8_linear_bf16(x, w, out=out, variant=variant)
    torch.cuda.synchronize()
    max_abs, mean_abs, p99_abs, cos = metrics(got, expected)
    tile = ops.select_fp8_linear_tile(m, n, k, variant)

    flashrt_us = measure(lambda: ops.fp8_linear_bf16(x, w, out=out, variant=variant), warmup, iters)
    eager_us = measure(lambda: ref_fn(x, w), warmup, iters)
    compile_us = None
    if compile_ref:
        try:
            compiled = torch.compile(ref_fn, fullgraph=True)
            compiled(x, w)
            torch.cuda.synchronize()
            compile_us = measure(lambda: compiled(x, w), warmup, iters)
        except Exception:
            compile_us = None

    return Result(
        shape=name,
        M=m,
        K=k,
        N=n,
        variant=variant,
        tile=tile,
        flashrt_us=flashrt_us,
        torch_eager_us=eager_us,
        torch_compile_us=compile_us,
        speedup_vs_eager=eager_us / flashrt_us,
        speedup_vs_compile=(compile_us / flashrt_us) if compile_us else None,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        cosine=cos,
        status="pass" if max_abs <= 0.5 and p99_abs <= 0.25 and cos >= 0.999 else "fail",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--compile-ref", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, _minor = torch.cuda.get_device_capability(0)
    if major < 12:
        raise SystemExit("fp8-gemm requires Blackwell/SM120 for this package")

    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    rows: list[Result] = []
    for name in MODES[args.mode]:
        shape = SHAPES[name]
        variants = [0]
        if shape[0] == 1:
            variants = [0, 4, 8, 16]
        for variant in variants:
            rows.append(bench_case(ops, name, shape, variant, args.warmup, args.iterations, args.compile_ref))

    payload = {"rows": [asdict(row) for row in rows]}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if any(row.status != "pass" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
