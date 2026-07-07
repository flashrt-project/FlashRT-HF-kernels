#!/usr/bin/env python3
"""Benchmark speculative-draft-primitives."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "speculative-draft-primitives" / "tests"))
from test_speculative_draft_primitives import load_installed_ops, load_source_ops  # noqa: E402


def time_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["headline", "full"], default="headline")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = [(16, 32000), (16, 248320)] if args.mode == "headline" else [
        (1, 1024),
        (4, 4096),
        (16, 32000),
        (16, 248320),
    ]
    print("rows,vocab,op,flashrt_us,torch_us,speedup,notes")
    for rows, vocab in shapes:
        logits = torch.randn((rows, vocab), device="cuda", dtype=torch.float32).to(torch.bfloat16)
        drafts = torch.argmax(logits.float(), dim=1)[: min(rows, 15)].contiguous()
        argmax_out = torch.empty((rows,), device=logits.device, dtype=torch.int64)
        accept_n = torch.empty((1,), device=logits.device, dtype=torch.int32)
        parts = 32 if vocab >= 131072 else (16 if vocab >= 65536 else (1 if vocab <= 4096 else 8))
        partial_vals = torch.empty((rows, parts), device=logits.device, dtype=torch.float32)
        partial_idx = torch.empty((rows, parts), device=logits.device, dtype=torch.int32)

        if hasattr(ops, "ops"):
            raw = ops.ops

            def flash_argmax():
                raw.argmax_bf16(logits, argmax_out)

            def flash_accept_partitioned():
                raw.accept_partitioned_bf16(
                    logits, drafts, argmax_out, accept_n, partial_vals, partial_idx, min(rows, 15), parts
                )
        else:
            def flash_argmax():
                ops.argmax_bf16(logits, out=argmax_out)

            def flash_accept_partitioned():
                ops.accept_partitioned_bf16(
                    logits,
                    drafts,
                    min(rows, 15),
                    parts,
                    argmax_out=argmax_out,
                    accept_n=accept_n,
                    partial_vals=partial_vals,
                    partial_idx=partial_idx,
                )

        flash_us = time_us(flash_argmax, args.warmup, args.iters)
        torch_us = time_us(lambda: torch.argmax(logits.float(), dim=1), args.warmup, args.iters)
        print(f"{rows},{vocab},argmax_bf16,{flash_us:.3f},{torch_us:.3f},{torch_us / flash_us:.2f},static-output")
        flash_us = time_us(flash_accept_partitioned, args.warmup, args.iters)
        print(f"{rows},{vocab},accept_partitioned_bf16,{flash_us:.3f},n/a,n/a,static-workspace parts={parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
