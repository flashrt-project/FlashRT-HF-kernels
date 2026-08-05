#!/usr/bin/env python3
"""Compare the static FA4 entry with PyTorch SDPA on GROOT shapes."""

import argparse
import importlib
import sys
from pathlib import Path

import torch


PACKAGE = Path(__file__).resolve().parents[1]


def timed_ms(fn, warmup=20, iterations=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact")
    args = parser.parse_args()
    root = Path(args.artifact) if args.artifact else PACKAGE / "torch-ext"
    sys.path.insert(0, str(root))
    ops = importlib.import_module("fa4_cute_runtime")
    for seq, hq, hk, dim, causal in [(41, 32, 32, 48, False), (277, 16, 16, 72, False), (1024, 16, 8, 128, True)]:
        q = torch.randn(1, seq, hq, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(1, seq, hk, dim, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        fa4_ms = timed_ms(lambda: ops.forward_static(q, k, v, out, causal=causal))
        sdpa_ms = timed_ms(lambda: torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=causal, enable_gqa=hq != hk))
        print(f"S={seq} H={hq}/{hk} D={dim} causal={causal} fa4_ms={fa4_ms:.6f} sdpa_ms={sdpa_ms:.6f} speedup={sdpa_ms/fa4_ms:.3f}x")


if __name__ == "__main__":
    main()
