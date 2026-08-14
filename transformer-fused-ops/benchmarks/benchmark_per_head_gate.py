#!/usr/bin/env python3
"""Benchmark the NHD per-head sigmoid gate producer."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "transformer-fused-ops" / "tests" / "test_transformer_fused_ops.py"


def load_ops(backend: str, artifact: str | None):
    spec = importlib.util.spec_from_file_location("transformer_ops_test", TEST)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.load_source_ops() if backend == "source" else module.load_installed_ops(artifact)


def time_cuda(fn, warmup: int, iters: int) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    ops = load_ops(args.backend, args.artifact)
    print("| Shape | Kernel us | Eager us | Speedup |")
    print("|---|---:|---:|---:|")
    for s, h, d in ((6144, 32, 128), (24576, 32, 128), (2688, 32, 64)):
        x = torch.randn((1, s, h, d), device="cuda", dtype=torch.bfloat16)
        gate = torch.randn((1, s, h), device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(x)

        def kernel():
            return ops.per_head_sigmoid_gate_bf16(x, gate, out=out)

        def eager():
            factor = (2.0 * torch.sigmoid(gate.float())).to(torch.bfloat16)
            return (x.float() * factor[..., None].float()).to(torch.bfloat16)

        got = kernel()
        ref = eager()
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        kernel_us = time_cuda(kernel, args.warmup, args.iters)
        eager_us = time_cuda(eager, args.warmup, args.iters)
        print(f"| S{s} H{h} D{d} | {kernel_us:.3f} | {eager_us:.3f} | {eager_us / kernel_us:.2f}x |")


if __name__ == "__main__":
    main()
