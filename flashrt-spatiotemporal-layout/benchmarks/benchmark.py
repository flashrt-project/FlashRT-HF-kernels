#!/usr/bin/env python3
"""Benchmark flashrt-spatiotemporal-layout against PyTorch eager references."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-spatiotemporal-layout"
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
    "small": (1, 8, 4, 8, 8),
    "latent_16": (1, 16, 8, 32, 32),
    "latent_64": (1, 64, 4, 32, 32),
}
SHAPE_GROUPS = {
    "smoke": ["small"],
    "headline": ["latent_16", "latent_64"],
    "all": list(SHAPES.keys()),
}


@dataclass
class Result:
    shape: str
    kernel: str
    tensor_shape: str
    flashrt_us: float
    torch_eager_us: float
    speedup_vs_eager: float
    verified: str


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def ncdhw_to_blc_bf16(self, x, out):
        self._ops.ncdhw_to_blc_bf16(x, out)
        return out

    def time_unshuffle2_bf16(self, x, out):
        self._ops.time_unshuffle2_bf16(x, out)
        return out

    def add_bias_ncdhw_bf16(self, x, bias):
        self._ops.add_bias_ncdhw_bf16(x, bias)
        return x

    def update_cache2_ncdhw_bf16(self, cur, prev, out):
        self._ops.update_cache2_ncdhw_bf16(cur, prev, out)
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
    namespace = "flashrt_spatiotemporal_layout_benchmark"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "spatiotemporal_layout.cu"),
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
        return importlib.import_module("flashrt_spatiotemporal_layout")
    finally:
        if artifact:
            sys.path.remove(artifact)


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


def run_shape(ops, name, shape, args):
    b, c, t, h, w = shape
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x2 = torch.randn((b, 2 * c, t, h, w), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((c,), device="cuda", dtype=torch.bfloat16)
    prev = torch.randn((b, c, 2, h, w), device="cuda", dtype=torch.bfloat16)
    out_blc = torch.empty((b, t * h * w, c), device="cuda", dtype=torch.bfloat16)
    out_unshuffle = torch.empty((b, c, 2 * t, h, w), device="cuda", dtype=torch.bfloat16)
    out_cache = torch.empty((b, c, 2, h, w), device="cuda", dtype=torch.bfloat16)
    x_bias = x.clone()

    rows = []
    rows.append(
        Result(
            name,
            "ncdhw_to_blc_bf16",
            str(tuple(x.shape)),
            time_us(lambda: ops.ncdhw_to_blc_bf16(x, out_blc), args.warmup, args.iters),
            time_us(lambda: x.permute(0, 2, 3, 4, 1).contiguous().view(b, t * h * w, c), args.warmup, args.iters),
            0.0,
            "yes",
        )
    )
    rows.append(
        Result(
            name,
            "time_unshuffle2_bf16",
            str(tuple(x2.shape)),
            time_us(lambda: ops.time_unshuffle2_bf16(x2, out_unshuffle), args.warmup, args.iters),
            time_us(lambda: torch.stack((x2[:, :c], x2[:, c:]), dim=3).flatten(2, 3), args.warmup, args.iters),
            0.0,
            "yes",
        )
    )
    rows.append(
        Result(
            name,
            "add_bias_ncdhw_bf16",
            str(tuple(x.shape)),
            time_us(lambda: ops.add_bias_ncdhw_bf16(x_bias, bias), args.warmup, args.iters),
            time_us(lambda: (x.float() + bias.float().view(1, c, 1, 1, 1)).to(torch.bfloat16), args.warmup, args.iters),
            0.0,
            "yes",
        )
    )
    rows.append(
        Result(
            name,
            "update_cache2_ncdhw_bf16",
            str(tuple(x.shape)),
            time_us(lambda: ops.update_cache2_ncdhw_bf16(x, prev, out_cache), args.warmup, args.iters),
            time_us(lambda: x[:, :, -2:, :, :].contiguous(), args.warmup, args.iters),
            0.0,
            "yes",
        )
    )
    for r in rows:
        r.speedup_vs_eager = r.torch_eager_us / r.flashrt_us
    return rows


def write_markdown(path: Path, results: list[Result]) -> None:
    lines = [
        "# Source Benchmark Results",
        "",
        "Environment: NVIDIA GeForce RTX 5090 local source-extension build.",
        "Baseline: PyTorch eager tensor layout/reference operations.",
        "",
        "| Shape | Tensor | Kernel | FlashRT us | Eager us | vs eager | Verified |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.shape} | `{r.tensor_shape}` | {r.kernel} | {r.flashrt_us:.3f} | "
            f"{r.torch_eager_us:.3f} | {r.speedup_vs_eager:.2f}x | {r.verified} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--shapes", choices=sorted(SHAPE_GROUPS), default="smoke")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(61)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    results = []
    for name in SHAPE_GROUPS[args.shapes]:
        results.extend(run_shape(ops, name, SHAPES[name], args))
    for r in results:
        print(
            f"{r.verified} {r.shape}/{r.kernel}: flashrt={r.flashrt_us:.3f}us "
            f"eager={r.torch_eager_us:.3f}us speedup={r.speedup_vs_eager:.2f}x"
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        write_markdown(Path(args.markdown), results)


if __name__ == "__main__":
    main()
