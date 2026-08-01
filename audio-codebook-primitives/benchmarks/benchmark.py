#!/usr/bin/env python3
"""Higgs delayed-codebook benchmark against eager/compile and raw op."""

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
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    if args.backend == "source":
        tests = Path(__file__).resolve().parents[1] / "tests"
        sys.path.insert(0, str(tests))
        from test_audio_codebook_primitives import load_source_ops
        ops = load_source_ops()
    else:
        if args.artifact:
            sys.path.insert(0, args.artifact)
        ops = importlib.import_module("audio_codebook_primitives")

    c, v, h, delay, boc = 8, 1026, 1024, 7, 1024
    logits = torch.randn((c, v), device="cuda", dtype=torch.bfloat16)
    codebook = torch.randn((c, v, h), device="cuda", dtype=torch.bfloat16)
    index = torch.arange(c, device="cuda")
    active = index <= delay
    boc_tensor = torch.full((c,), boc, device="cuda", dtype=torch.int64)

    def eager():
        codes = torch.where(active, logits.argmax(dim=1), boc_tensor)
        embedding = codebook[index, codes].float().sum(dim=0).bfloat16()
        return codes, embedding

    compiled = torch.compile(eager, fullgraph=True)
    codes = torch.empty(c, device="cuda", dtype=torch.int64)
    embedding = torch.empty(h, device="cuda", dtype=torch.bfloat16)

    def wrapper():
        return ops.delayed_codebook_argmax_embed_bf16(
            logits, codebook, delay=delay, boc=boc,
            codes=codes, embedding=embedding
        )

    namespace = ops.ops

    def raw():
        namespace.delayed_codebook_argmax_embed_bf16(
            logits, codebook, delay, boc, codes, embedding
        )

    expected = eager()
    actual = wrapper()
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    rows = {
        "torch_eager_us": elapsed_us(eager, args.warmup, args.iterations),
        "torch_compile_us": elapsed_us(compiled, args.warmup, args.iterations),
        "hub_wrapper_us": elapsed_us(wrapper, args.warmup, args.iterations),
        "raw_native_op_us": elapsed_us(raw, args.warmup, args.iterations),
    }
    for name, value in rows.items():
        print(f"{name}={value:.3f}")


if __name__ == "__main__":
    main()
