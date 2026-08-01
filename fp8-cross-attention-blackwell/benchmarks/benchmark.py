#!/usr/bin/env python3
"""Benchmark FP8 GQA cross-attention against eager/compile and raw native op."""

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


def quantize(x):
    scale = max(float(x.abs().max()) / 448.0, 1e-6)
    return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--sq", type=int, default=786)
    parser.add_argument("--sk", type=int, default=7984)
    parser.add_argument("--hq", type=int, default=28)
    parser.add_argument("--hkv", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    tests = Path(__file__).resolve().parents[1] / "tests"
    sys.path.insert(0, str(tests))
    from test_fp8_cross_attention_blackwell import reference
    if args.backend == "source":
        from test_fp8_cross_attention_blackwell import load_source_ops
        ops = load_source_ops()
    else:
        if args.artifact:
            sys.path.insert(0, args.artifact)
        ops = importlib.import_module("fp8_cross_attention_blackwell")

    q, qs = quantize(torch.randn((1, args.sq, args.hq, 128), device="cuda") * 0.2)
    k, ks = quantize(torch.randn((1, args.sk, args.hkv, 128), device="cuda") * 0.2)
    v, vs = quantize(torch.randn((1, args.sk, args.hkv, 128), device="cuda") * 0.2)
    output = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(
        (1, args.hq, (args.sq + 127) // 128 * 128),
        device="cuda", dtype=torch.float32,
    )
    workspace = torch.empty(4 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    def eager():
        return reference(q, k, v, qs, ks, vs)

    compiled = torch.compile(eager, fullgraph=True)

    def wrapper():
        return ops.fp8_gqa_cross_attention_bf16(
            q, k, v, query_scale=qs, key_scale=ks, value_scale=vs,
            output=output, lse=lse, workspace=workspace,
        )

    def raw():
        ops.ops.fp8_gqa_cross_attention_bf16_out(
            q, k, v, qs, ks, vs, output, lse, workspace
        )

    expected = eager()
    actual = wrapper()
    diff = (actual.float() - expected.float()).abs()
    assert float(diff.mean()) <= 0.0005
    rows = {
        "torch_eager_us": elapsed_us(eager, args.warmup, args.iterations),
        "torch_compile_us": elapsed_us(compiled, args.warmup, args.iterations),
        "hub_wrapper_us": elapsed_us(wrapper, args.warmup, args.iterations),
        "raw_native_op_us": elapsed_us(raw, args.warmup, args.iterations),
    }
    print(f"shape=Sq{args.sq}_Sk{args.sk}_Hq{args.hq}_Hkv{args.hkv}_D128")
    for name, value in rows.items():
        print(f"{name}={value:.3f}")


if __name__ == "__main__":
    main()
