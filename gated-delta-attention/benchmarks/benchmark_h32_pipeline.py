#!/usr/bin/env python3
"""Benchmark the generic H32 producer/chunk path against native staging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def elapsed_us(fn, warmup: int, iterations: int) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--sequences", default="1,4,64")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    tests = Path(__file__).resolve().parents[1] / "tests"
    sys.path.insert(0, str(tests))
    from test_gated_delta_attention import load_installed_ops, load_source_ops, make_conv_inputs_h

    if args.backend == "source":
        ops = load_source_ops()
    else:
        ops = load_installed_ops(args.artifact)

    Hv, Hk, D = 32, 16, 128
    for S in (int(x) for x in args.sequences.split(",")):
        conv, a, b, neg, dt, initial = make_conv_inputs_h(S, Hv, Hk, 550000 + S)
        fused_state = initial.clone()
        staged_state = initial.clone()
        out = torch.empty((S, Hv, D), device="cuda", dtype=torch.bfloat16)

        def fused_native():
            fused_state.copy_(initial)
            return ops.chunk_from_conv_h(
                conv, a, b, neg, dt, fused_state, Hv, Hk, out=out
            )

        def staged_native():
            staged_state.copy_(initial)
            q, k, v = ops.split_broadcast_h(conv, Hv, Hk)
            g, beta = ops.gating_h(a, b, neg, dt, Hv)
            return ops.chunk(q, k, v, g, beta, staged_state, smem=True)

        got = fused_native().clone()
        ref = staged_native().clone()
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        torch.testing.assert_close(fused_state, staged_state, rtol=0, atol=0)
        fused_us = elapsed_us(fused_native, args.warmup, args.iterations)
        staged_us = elapsed_us(staged_native, args.warmup, args.iterations)
        print(
            f"S={S} Hv={Hv} Hk={Hk} D={D} "
            f"fused_native_us={fused_us:.3f} staged_native_us={staged_us:.3f} "
            f"speedup={staged_us / fused_us:.2f}x exact=True"
        )


if __name__ == "__main__":
    main()
