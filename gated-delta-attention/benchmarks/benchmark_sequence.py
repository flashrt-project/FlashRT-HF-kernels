#!/usr/bin/env python3
"""Benchmark the full recurrent scan against eager, compile, and native paths."""

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
    parser.add_argument("--sequence", type=int, default=65)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    tests = Path(__file__).resolve().parents[1] / "tests"
    sys.path.insert(0, str(tests))
    from test_gated_delta_attention import (
        InstalledOps,
        load_source_ops,
        ref_sequence_f32state,
    )

    if args.backend == "source":
        ops = load_source_ops()
        raw_ops = ops._ops
    else:
        if args.artifact:
            sys.path.insert(0, args.artifact)
        ops = InstalledOps(importlib.import_module("gated_delta_attention"))
        raw_ops = ops._mod.ops

    s, h, d = args.sequence, args.heads, 128
    q = (torch.randn((s, h, d), device="cuda") * 0.05).bfloat16()
    k = (torch.randn((s, h, d), device="cuda") * 0.05).bfloat16()
    v = (torch.randn((s, h, d), device="cuda") * 0.05).bfloat16()
    g = (torch.randn((s, h), device="cuda") * 0.02).bfloat16()
    beta = torch.sigmoid(torch.randn((s, h), device="cuda")).bfloat16()
    state_initial = torch.zeros((h, d, d), device="cuda", dtype=torch.bfloat16)
    state = state_initial.clone()
    legacy_state = state_initial.unsqueeze(0).clone()
    output = torch.empty_like(q)

    def eager():
        return ref_sequence_f32state(q, k, v, g, beta, state_initial)

    compiled = torch.compile(eager, fullgraph=True)

    def wrapper():
        state.copy_(state_initial)
        return ops.sequence(q, k, v, g, beta, state, out=output)

    def raw_native_sequence():
        state.copy_(state_initial)
        raw_ops.gated_delta_recurrent_sequence_bf16(
            q, k, v, g, beta, state, output, True
        )

    def legacy_per_token_native():
        legacy_state.copy_(state_initial.unsqueeze(0))
        for index in range(s):
            ops.recurrent(
                q[index:index + 1], k[index:index + 1], v[index:index + 1],
                g[index:index + 1], beta[index:index + 1], legacy_state,
            )

    expected, _ = eager()
    actual = wrapper()
    diff = (actual.float() - expected.float()).abs()
    assert float(torch.quantile(diff.flatten(), 0.99)) <= 0.03125
    values = {
        "torch_eager_us": elapsed_us(eager, args.warmup, args.iterations),
        "torch_compile_us": elapsed_us(compiled, args.warmup, args.iterations),
        "legacy_per_token_native_us": elapsed_us(
            legacy_per_token_native, args.warmup, args.iterations
        ),
        "hub_wrapper_us": elapsed_us(wrapper, args.warmup, args.iterations),
        "raw_native_sequence_us": elapsed_us(
            raw_native_sequence, args.warmup, args.iterations
        ),
    }
    print(f"shape=S{s}_H{h}_D{d}")
    for name, value in values.items():
        print(f"{name}={value:.3f}")


if __name__ == "__main__":
    main()
