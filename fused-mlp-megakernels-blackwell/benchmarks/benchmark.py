#!/usr/bin/env python3
"""Benchmark the fused GeGLU region against eager, compile, and raw native op."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _apply_mem_cap(max_mem_gb: float = 30.0) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


def elapsed_us(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--m", type=int, default=768)
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)

    if args.backend == "source":
        tests = Path(__file__).resolve().parents[1] / "tests"
        sys.path.insert(0, str(tests))
        from test_fused_mlp_megakernels_blackwell import load_source_ops
        ops = load_source_ops()
    else:
        if args.artifact:
            sys.path.insert(0, args.artifact)
        ops = importlib.import_module("fused_mlp_megakernels_blackwell")

    x = (torch.randn((args.m, args.k), device="cuda") * 0.15).half()
    gate = (torch.randn((args.n, args.k), device="cuda") * 0.05).half()
    up = (torch.randn((args.n, args.k), device="cuda") * 0.05).half()
    scratch = torch.empty((args.m, args.n), device="cuda", dtype=torch.float16)
    output = torch.empty_like(scratch)

    def eager():
        return F.gelu(x @ gate.t(), approximate="tanh") * (x @ up.t())

    compiled = torch.compile(eager, fullgraph=True)

    def wrapper():
        return ops.fp16_geglu_fused(
            x, gate, up, gate_scratch=scratch, output=output
        )

    def raw():
        ops.ops.fp16_geglu_fused_out(x, gate, up, scratch, output)

    expected = eager()
    actual = wrapper()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    rows = {
        "torch_eager_us": elapsed_us(eager, args.warmup, args.iterations),
        "torch_compile_us": elapsed_us(compiled, args.warmup, args.iterations),
        "hub_wrapper_us": elapsed_us(wrapper, args.warmup, args.iterations),
        "raw_native_op_us": elapsed_us(raw, args.warmup, args.iterations),
    }
    print(f"shape=M{args.m}_N{args.n}_K{args.k}")
    for name, value in rows.items():
        print(f"{name}={value:.3f}")


if __name__ == "__main__":
    main()
