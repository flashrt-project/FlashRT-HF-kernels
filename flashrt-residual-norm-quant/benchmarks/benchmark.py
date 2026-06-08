#!/usr/bin/env python3
"""Benchmark flashrt-residual-norm-quant against PyTorch eager references."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-residual-norm-quant"
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
    "pi05_decoder": (10, 1024),
    "pi05_vision": (512, 1152),
    "groot_vl": (1024, 2048),
    "video_prefill": (2520, 2048),
}
SHAPE_GROUPS = {
    "smoke": ["pi05_decoder"],
    "headline": ["pi05_decoder", "pi05_vision", "groot_vl"],
    "all": list(SHAPES.keys()),
}


@dataclass
class Result:
    shape: str
    rows: int
    dim: int
    kernel: str
    flashrt_us: float
    torch_eager_us: float
    speedup_vs_eager: float
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    status: str


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def rms_norm_quant_fp8_static_bf16(self, x, weight, scale, eps=1e-6, out=None):
        if out is None:
            out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self._ops.rms_norm_quant_fp8_static_bf16(x, weight, scale, float(eps), out)
        return out

    def residual_add_rms_norm_quant_fp8_static_bf16(
        self, residual, x, weight, scale, eps=1e-6, out=None
    ):
        if out is None:
            out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self._ops.residual_add_rms_norm_quant_fp8_static_bf16(
            residual, x, weight, scale, float(eps), out
        )
        return out


def _preload_cublaslt() -> None:
    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    library = ctypes.util.find_library("cublasLt")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "flashrt_residual_norm_quant_benchmark"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "residual_norm_quant.cu"),
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
        return importlib.import_module("flashrt_residual_norm_quant")
    finally:
        if artifact:
            sys.path.remove(artifact)


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)


def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.rsqrt(torch.mean(x.float() * x.float(), dim=1, keepdim=True) + eps)
    return x.float() * rms * weight.float()


def torch_rms_norm_quant(x, weight, scale, eps) -> torch.Tensor:
    return quantize_fp8(torch_rms_norm(x, weight, eps), scale)


def torch_residual_add_rms_norm_quant(residual, x, weight, scale, eps) -> torch.Tensor:
    added = residual.float() + x.float()
    residual.copy_(added.to(torch.bfloat16))
    rms = torch.rsqrt(torch.mean(added * added, dim=1, keepdim=True) + eps)
    return quantize_fp8(residual.float() * rms * weight.float(), scale)


def make_case(rows: int, dim: int):
    x = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn((dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    out = torch.empty((rows, dim), device="cuda", dtype=torch.float8_e4m3fn)
    return x, residual, weight, scale, out


def time_us(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) * 1000.0 / iters


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def metrics(got: torch.Tensor, expected: torch.Tensor):
    diff = (got.float() - expected.float()).abs().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        got.float().flatten(), expected.float().flatten(), dim=0
    )
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(percentile(diff, 0.99).item()),
        "cosine": float(cosine.item()),
    }


def run_one(ops, name: str, rows: int, dim: int, args) -> list[Result]:
    x, residual, weight, scale, out = make_case(rows, dim)
    eps = args.eps
    results = []

    got = ops.rms_norm_quant_fp8_static_bf16(x, weight, scale, eps, out)
    expected = torch_rms_norm_quant(x, weight, scale, eps)
    m = metrics(got, expected)
    kernel_us = time_us(
        lambda: ops.rms_norm_quant_fp8_static_bf16(x, weight, scale, eps, out),
        args.warmup,
        args.iters,
    )
    torch_us = time_us(lambda: torch_rms_norm_quant(x, weight, scale, eps), args.warmup, args.iters)
    results.append(
        Result(
            shape=name,
            rows=rows,
            dim=dim,
            kernel="rms_norm_quant_fp8_static_bf16",
            flashrt_us=kernel_us,
            torch_eager_us=torch_us,
            speedup_vs_eager=torch_us / kernel_us,
            status="PASS" if m["p99_abs"] <= args.p99_abs_limit else "FAIL",
            **m,
        )
    )

    residual0 = residual.clone()
    residual_kernel = residual0.clone()
    got = ops.residual_add_rms_norm_quant_fp8_static_bf16(
        residual_kernel, x, weight, scale, eps, out
    )
    residual_ref = residual0.clone()
    expected = torch_residual_add_rms_norm_quant(residual_ref, x, weight, scale, eps)
    m = metrics(got, expected)
    residual_kernel = residual0.clone()
    residual_ref = residual0.clone()
    kernel_us = time_us(
        lambda: ops.residual_add_rms_norm_quant_fp8_static_bf16(
            residual_kernel, x, weight, scale, eps, out
        ),
        args.warmup,
        args.iters,
    )
    torch_us = time_us(
        lambda: torch_residual_add_rms_norm_quant(residual_ref, x, weight, scale, eps),
        args.warmup,
        args.iters,
    )
    results.append(
        Result(
            shape=name,
            rows=rows,
            dim=dim,
            kernel="residual_add_rms_norm_quant_fp8_static_bf16",
            flashrt_us=kernel_us,
            torch_eager_us=torch_us,
            speedup_vs_eager=torch_us / kernel_us,
            status="PASS" if m["p99_abs"] <= args.p99_abs_limit else "FAIL",
            **m,
        )
    )
    return results


def write_markdown(path: Path, results: list[Result]) -> None:
    lines = [
        "| Shape | Rows,Dim | Kernel | FlashRT us | Eager us | vs eager | Max abs | Mean abs | P99 abs | Cosine | Status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.shape} | {r.rows},{r.dim} | {r.kernel} | {r.flashrt_us:.3f} | "
            f"{r.torch_eager_us:.3f} | {r.speedup_vs_eager:.2f}x | "
            f"{r.max_abs:.6f} | {r.mean_abs:.6f} | {r.p99_abs:.6f} | "
            f"{r.cosine:.8f} | {r.status} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--shapes", choices=sorted(SHAPE_GROUPS), default="smoke")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--p99-abs-limit", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(29)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    results = []
    for name in SHAPE_GROUPS[args.shapes]:
        rows, dim = SHAPES[name]
        results.extend(run_one(ops, name, rows, dim, args))

    for r in results:
        print(
            f"{r.status} {r.shape}/{r.kernel}: flashrt={r.flashrt_us:.3f}us "
            f"eager={r.torch_eager_us:.3f}us speedup={r.speedup_vs_eager:.2f}x "
            f"p99_abs={r.p99_abs:.6f} cosine={r.cosine:.8f}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        write_markdown(Path(args.markdown), results)

    if any(r.status != "PASS" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
