#!/usr/bin/env python3
"""Benchmark fp8-kv-attention against a PyTorch FP8-dequant reference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import sys

TESTS = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(TESTS))
from test_fp8_kv_attention import SHAPES, SourceOps, load_installed_ops, load_source_ops, make_inputs, reference  # noqa: E402


MODES = {
    "smoke": ["decode_128"],
    "headline": ["decode_1024", "verify4_1024", "verify8_4096"],
    "full": ["decode_128", "decode_1024", "verify4_1024", "verify8_4096"],
}


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
    return float(start.elapsed_time(end) * 1000.0 / iters)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    rows = []
    for name in MODES[args.mode]:
        q_seq, kv_seq = SHAPES[name]
        q, k, v = make_inputs(q_seq, kv_seq, seed=3000 + q_seq * 17 + kv_seq)
        if isinstance(ops, SourceOps):
            def kernel_call():
                return ops.xqa_bf16_fp8kv(q, k, v, kv_seq)
        else:
            pages = k.shape[0]
            page_table = ops.default_page_table(pages, device=q.device)
            seq_lens = torch.tensor([[kv_seq]], device=q.device, dtype=torch.int32)
            mask = ops.causal_spec_mask(q_seq, device=q.device)
            sem, scratch = ops.allocate_workspace(q_seq=q_seq, device=q.device)
            out = torch.empty_like(q)

            def kernel_call():
                return ops.xqa_bf16_fp8kv(
                    q, k, v, page_table, seq_lens, mask,
                    out=out, semaphores=sem, scratch=scratch,
                    max_seq_len=pages * 128,
                )

        def ref_call():
            return reference(q, k, v, kv_seq)

        kernel_us = time_cuda(kernel_call, args.warmup, args.iters)
        ref_us = time_cuda(ref_call, max(2, args.warmup // 5), max(5, args.iters // 10))
        rows.append(
            {
                "shape": name,
                "q_seq": q_seq,
                "kv_seq": kv_seq,
                "kernel_us": kernel_us,
                "torch_reference_us": ref_us,
                "speedup": ref_us / kernel_us,
            }
        )
        print(f"{name}: kernel={kernel_us:.3f}us ref={ref_us:.3f}us speedup={ref_us / kernel_us:.2f}x")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
