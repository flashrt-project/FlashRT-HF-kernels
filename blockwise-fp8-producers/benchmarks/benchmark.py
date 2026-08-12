#!/usr/bin/env python3
"""Benchmark blockwise FP8 producer APIs."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "blockwise-fp8-producers" / "tests"))
from test_blockwise_fp8_producers import load_source_ops  # noqa: E402


def _apply_mem_cap(max_mem_gb: float = 30.0) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


def load_ops(backend: str, artifact: str | None):
    if backend == "source":
        return load_source_ops()
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("blockwise_fp8_producers")
    finally:
        if artifact:
            sys.path.remove(artifact)


def reference(kind, x, weight, bias):
    if kind == "layer_norm":
        produced = F.layer_norm(
            x.float(), (x.shape[1],), weight.float(), bias.float(), 1e-6
        )
    elif kind == "rms_norm":
        produced = (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        )
    elif kind == "gelu_bias":
        produced = F.gelu(x.float() + bias.float(), approximate="tanh")
    else:
        produced = x.float()
    blocks = produced.reshape(produced.shape[0], produced.shape[1] // 128, 128)
    scale = torch.clamp(blocks.abs().amax(-1) / 448.0, min=1.0e-12)
    quantized = torch.clamp(
        blocks / scale.unsqueeze(-1), -448.0, 448.0
    ).to(torch.float8_e4m3fn)
    return quantized.reshape_as(x), scale


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["headline", "full"], default="headline")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)
    ops = load_ops(args.backend, args.artifact)
    shapes = [(51, 4096), (277, 9216), (1024, 1152)]
    if args.mode == "full":
        shapes = [(1, 4096), (17, 1152), (51, 4096), (65, 4352), (277, 9216), (1024, 1152)]

    print("kind,shape,artifact_us,eager_us,compile_us,eager_speedup,compile_speedup")
    for rows, dim in shapes:
        x = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((dim,), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn((dim,), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty((rows, dim // 128), device="cuda", dtype=torch.float32)
        for kind in ("quantize", "layer_norm", "rms_norm", "gelu_bias"):
            if kind == "quantize":
                artifact_fn = lambda: ops.quantize_fp8_block128_bf16(
                    x, output=output, scale=scale
                )
            elif kind == "layer_norm":
                artifact_fn = lambda: ops.layer_norm_fp8_block128_bf16(
                    x, weight, bias, output=output, scale=scale
                )
            elif kind == "rms_norm":
                artifact_fn = lambda: ops.rms_norm_fp8_block128_bf16(
                    x, weight, output=output, scale=scale
                )
            else:
                artifact_fn = lambda: ops.gelu_tanh_bias_fp8_block128_bf16(
                    x, bias, output=output, scale=scale
                )
            eager_fn = lambda: reference(kind, x, weight, bias)
            torch._dynamo.reset()
            compiled = torch.compile(
                lambda a, w, b: reference(kind, a, w, b), fullgraph=True
            )
            compiled_fn = lambda: compiled(x, weight, bias)
            artifact_us = time_us(artifact_fn, args.warmup, args.iters)
            eager_us = time_us(eager_fn, max(10, args.warmup // 2), max(50, args.iters // 2))
            compile_us = time_us(compiled_fn, max(10, args.warmup // 2), max(50, args.iters // 2))
            print(
                f"{kind},{rows}x{dim},{artifact_us:.3f},{eager_us:.3f},"
                f"{compile_us:.3f},{eager_us/artifact_us:.2f}x,"
                f"{compile_us/artifact_us:.2f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
