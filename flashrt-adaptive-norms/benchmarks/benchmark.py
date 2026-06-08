#!/usr/bin/env python3
"""Benchmark flashrt-adaptive-norms against PyTorch eager references."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-adaptive-norms"
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
    "small": (64, 1024),
    "vla_2k": (2520, 3072),
    "vla_4k": (4096, 3072),
}
SHAPE_GROUPS = {
    "smoke": ["small"],
    "headline": ["vla_2k"],
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
    p99_abs: float
    cosine: float
    status: str


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def ada_rms_norm_style_bf16(self, x, weight, style, eps, out, gate_out):
        self._ops.ada_rms_norm_style_bf16(x, weight, style, float(eps), out, gate_out)
        return out, gate_out

    def gate_residual_ada_norm_fp8_static_bf16(self, residual, x, gate, weight, style, scale, eps, out, gate_out):
        self._ops.gate_residual_ada_norm_fp8_static_bf16(
            residual, x, gate, weight, style, scale, float(eps), out, gate_out
        )
        return residual, out, gate_out


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
    namespace = "flashrt_adaptive_norms_benchmark"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "adaptive_norms.cu"),
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
        return importlib.import_module("flashrt_adaptive_norms")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_case(rows: int, dim: int):
    x = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    gate = torch.randn_like(x)
    weight = (1.0 + 0.1 * torch.randn((dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    style = (0.05 * torch.randn((rows, 3 * dim), device="cuda", dtype=torch.bfloat16)).contiguous()
    scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    return x, residual, gate, weight, style, scale


def rms_norm(x, weight, eps):
    rms = torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps)
    return x.float() * rms * weight.float()


def ref_ada(x, weight, style, eps):
    dim = x.shape[1]
    normed = rms_norm(x, weight, eps)
    y = normed * (1.0 + style[:, :dim].float()) + style[:, dim : 2 * dim].float()
    return y.to(torch.bfloat16), style[:, 2 * dim :].contiguous().to(torch.bfloat16)


def ref_fused(residual, x, gate, weight, style, scale, eps):
    updated = (residual.float() + x.float() * gate.float()).to(torch.bfloat16)
    dim = updated.shape[1]
    normed = rms_norm(updated, weight, eps)
    y = (normed * (1.0 + style[:, :dim].float()) + style[:, dim : 2 * dim].float()) / scale.float().reshape(())
    return updated, y.to(torch.float8_e4m3fn), style[:, 2 * dim :].contiguous().to(torch.bfloat16)


def time_us(fn, warmup, iters):
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


def percentile(x, q):
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def metrics(got, expected):
    diff = (got.float() - expected.float()).abs().flatten()
    return float(percentile(diff, 0.99).item()), float(
        torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item()
    )


def run_one(ops, name, shape, args):
    rows, dim = shape
    x, residual, gate, weight, style, scale = make_case(rows, dim)
    out = torch.empty_like(x)
    gate_out = torch.empty_like(x)
    fp8_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    fused_gate_out = torch.empty_like(x)
    ada_got, _ = ops.ada_rms_norm_style_bf16(x, weight, style, args.eps, out, gate_out)
    ada_exp, _ = ref_ada(x, weight, style, args.eps)
    ada_p99, ada_cos = metrics(ada_got, ada_exp)
    ada_flashrt_us = time_us(lambda: ops.ada_rms_norm_style_bf16(x, weight, style, args.eps, out, gate_out), args.warmup, args.iters)
    ada_eager_us = time_us(lambda: ref_ada(x, weight, style, args.eps), args.warmup, args.iters)

    residual_work = residual.clone()
    fused_got = ops.gate_residual_ada_norm_fp8_static_bf16(
        residual_work, x, gate, weight, style, scale, args.eps, fp8_out, fused_gate_out
    )[1]
    _, fused_exp, _ = ref_fused(residual, x, gate, weight, style, scale, args.eps)
    fused_p99, fused_cos = metrics(fused_got.float(), fused_exp.float())
    fused_flashrt_us = time_us(
        lambda: ops.gate_residual_ada_norm_fp8_static_bf16(
            residual_work, x, gate, weight, style, scale, args.eps, fp8_out, fused_gate_out
        ),
        args.warmup,
        args.iters,
    )
    fused_eager_us = time_us(lambda: ref_fused(residual, x, gate, weight, style, scale, args.eps), args.warmup, args.iters)

    return [
        Result(name, rows, dim, "ada_rms_norm_style_bf16", ada_flashrt_us, ada_eager_us, ada_eager_us / ada_flashrt_us, ada_p99, ada_cos, "PASS"),
        Result(name, rows, dim, "gate_residual_ada_norm_fp8_static_bf16", fused_flashrt_us, fused_eager_us, fused_eager_us / fused_flashrt_us, fused_p99, fused_cos, "PASS"),
    ]


def write_markdown(path: Path, results: list[Result]) -> None:
    lines = [
        "# Source Benchmark Results",
        "",
        "Environment: NVIDIA GeForce RTX 5090 local source-extension build.",
        "Baseline: PyTorch eager tensor reference with matching BF16/FP8 math contract.",
        "",
        "| Shape | Rows,Dim | Kernel | FlashRT us | Eager us | vs eager | p99 abs | Cosine | Status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.shape} | {r.rows},{r.dim} | {r.kernel} | {r.flashrt_us:.3f} | "
            f"{r.torch_eager_us:.3f} | {r.speedup_vs_eager:.2f}x | {r.p99_abs:.6f} | "
            f"{r.cosine:.8f} | {r.status} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--shapes", choices=sorted(SHAPE_GROUPS), default="smoke")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(53)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    results = []
    for name in SHAPE_GROUPS[args.shapes]:
        results.extend(run_one(ops, name, SHAPES[name], args))
    for r in results:
        print(
            f"{r.status} {r.shape}/{r.kernel}: flashrt={r.flashrt_us:.3f}us "
            f"eager={r.torch_eager_us:.3f}us speedup={r.speedup_vs_eager:.2f}x p99={r.p99_abs:.6f}"
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        write_markdown(Path(args.markdown), results)


if __name__ == "__main__":
    main()
