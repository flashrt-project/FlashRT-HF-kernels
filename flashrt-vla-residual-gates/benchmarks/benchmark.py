#!/usr/bin/env python3
"""Benchmark flashrt-vla-residual-gates against PyTorch eager."""

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
PACKAGE = ROOT / "flashrt-vla-residual-gates"
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
    "small": ((64, 8, 4), 1024),
    "vla_2k": ((2520, 16, 16), 3072),
    "vla_4k": ((4096, 16, 16), 3072),
}
SHAPE_GROUPS = {
    "smoke": ["small"],
    "headline": ["vla_2k"],
    "all": list(SHAPES.keys()),
}


@dataclass
class Result:
    shape: str
    rows: str
    dim: int
    flashrt_us: float
    torch_eager_us: float
    speedup_vs_eager: float
    p99_abs: float
    cosine: float
    status: str


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def joint3_bias_gate_residual_action_nobias_bf16(self, *args):
        self._ops.joint3_bias_gate_residual_action_nobias_bf16(*args)


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
    namespace = "flashrt_vla_residual_gates_benchmark"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "residual_gates.cu"),
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
        return importlib.import_module("flashrt_vla_residual_gates")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_segment(rows: int, dim: int):
    residual = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn_like(residual)
    bias = (0.02 * torch.randn((dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    out = torch.empty_like(residual)
    return residual, x, bias, gate, out


def make_case(rows: tuple[int, int, int], dim: int):
    v = make_segment(rows[0], dim)
    a = make_segment(rows[1], dim)
    u_residual = torch.randn((rows[2], dim), device="cuda", dtype=torch.bfloat16)
    u_x = torch.randn_like(u_residual)
    u_out = torch.empty_like(u_residual)
    return v, a, (u_residual, u_x, u_out)


def torch_ref(v, a, u):
    v_ref = (v[0].float() + (v[1].float() + v[2].float().view(1, -1)) * v[3].float()).to(torch.bfloat16)
    a_ref = (a[0].float() + a[1].float() * a[3].float()).to(torch.bfloat16)
    u_ref = (u[0].float() + u[1].float()).to(torch.bfloat16)
    return v_ref, a_ref, u_ref


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


def metrics(got_parts, expected_parts):
    got = torch.cat([x.flatten() for x in got_parts]).float()
    expected = torch.cat([x.flatten() for x in expected_parts]).float()
    diff = (got - expected).abs()
    return float(percentile(diff, 0.99).item()), float(
        torch.nn.functional.cosine_similarity(got, expected, dim=0).item()
    )


def run_one(ops, name: str, rows: tuple[int, int, int], dim: int, args) -> Result:
    v, a, u = make_case(rows, dim)
    ops.joint3_bias_gate_residual_action_nobias_bf16(
        v[0], v[1], v[2], v[3], v[4],
        a[0], a[1], a[3], a[4],
        u[0], u[1], u[2],
    )
    expected = torch_ref(v, a, u)
    p99_abs, cosine = metrics((v[4], a[4], u[2]), expected)
    flashrt_us = time_us(
        lambda: ops.joint3_bias_gate_residual_action_nobias_bf16(
            v[0], v[1], v[2], v[3], v[4],
            a[0], a[1], a[3], a[4],
            u[0], u[1], u[2],
        ),
        args.warmup,
        args.iters,
    )
    eager_us = time_us(lambda: torch_ref(v, a, u), args.warmup, args.iters)
    status = "PASS" if p99_abs <= args.p99_abs_limit else "FAIL"
    return Result(
        shape=name,
        rows=f"{rows[0]},{rows[1]},{rows[2]}",
        dim=dim,
        flashrt_us=flashrt_us,
        torch_eager_us=eager_us,
        speedup_vs_eager=eager_us / flashrt_us,
        p99_abs=p99_abs,
        cosine=cosine,
        status=status,
    )


def write_markdown(path: Path, results: list[Result]) -> None:
    lines = [
        "# Source Benchmark Results",
        "",
        "Environment: NVIDIA GeForce RTX 5090 local source-extension build.",
        "Baseline: PyTorch eager tensor postprocess chain with matching BF16 math.",
        "",
        "| Shape | V,A,U rows | Dim | FlashRT us | Eager us | vs eager | p99 abs | Cosine | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.shape} | {r.rows} | {r.dim} | {r.flashrt_us:.3f} | "
            f"{r.torch_eager_us:.3f} | {r.speedup_vs_eager:.2f}x | "
            f"{r.p99_abs:.6f} | {r.cosine:.8f} | {r.status} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--shapes", choices=sorted(SHAPE_GROUPS), default="smoke")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--p99-abs-limit", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(43)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    results = [run_one(ops, name, *SHAPES[name], args) for name in SHAPE_GROUPS[args.shapes]]
    for r in results:
        print(
            f"{r.status} {r.shape}: flashrt={r.flashrt_us:.3f}us eager={r.torch_eager_us:.3f}us "
            f"speedup={r.speedup_vs_eager:.2f}x p99={r.p99_abs:.6f}"
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
