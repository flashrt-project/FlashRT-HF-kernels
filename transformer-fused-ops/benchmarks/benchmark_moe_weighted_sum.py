#!/usr/bin/env python3
"""Benchmark MoE routed-row reduction against eager, compile, and raw native op."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch


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
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2112)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    if args.backend == "source":
        tests = Path(__file__).resolve().parents[1] / "tests"
        sys.path.insert(0, str(tests))
        from test_transformer_fused_ops import load_source_ops
        ops = load_source_ops()
    else:
        if args.artifact:
            sys.path.insert(0, args.artifact)
        ops = importlib.import_module("transformer_fused_ops")

    rows = args.tokens * args.topk + 5
    expert = torch.randn((rows, args.stride), device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(
        0, rows, (args.tokens, args.topk), device="cuda", dtype=torch.int32
    )
    weights = torch.softmax(
        torch.randn((args.tokens, args.topk), device="cuda"), dim=-1
    )
    output = torch.empty((args.tokens, args.hidden), device="cuda", dtype=torch.float32)

    def eager():
        return (
            expert[indices.long(), :args.hidden].float() * weights[..., None]
        ).sum(dim=1)

    compiled = torch.compile(eager, fullgraph=True)

    def wrapper():
        return ops.moe_weighted_sum_bf16_to_fp32(
            expert, indices, weights, hidden=args.hidden, out=output
        )

    def raw():
        ops.ops.moe_weighted_sum_bf16_to_fp32(expert, indices, weights, output)

    torch.testing.assert_close(wrapper(), eager(), rtol=0, atol=1e-5)
    values = {
        "torch_eager_us": elapsed_us(eager, args.warmup, args.iterations),
        "torch_compile_us": elapsed_us(compiled, args.warmup, args.iterations),
        "hub_wrapper_us": elapsed_us(wrapper, args.warmup, args.iterations),
        "raw_native_op_us": elapsed_us(raw, args.warmup, args.iterations),
    }
    print(
        f"shape=tokens{args.tokens}_topk{args.topk}_hidden{args.hidden}_stride{args.stride}"
    )
    for name, value in values.items():
        print(f"{name}={value:.3f}")


if __name__ == "__main__":
    main()
