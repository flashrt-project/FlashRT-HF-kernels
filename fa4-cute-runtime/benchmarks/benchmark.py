#!/usr/bin/env python3
"""Compare static FA4 with PyTorch SDPA on qualified model shapes."""

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
    cases = [
        ("groot-dit", 41, 41, None, 32, 32, 48, False),
        ("pi05-vision", 277, 277, None, 16, 16, 72, False),
        ("groot-llm", 1024, 1024, None, 16, 8, 128, True),
        ("pi05-encoder-dense", 320, 320, None, 8, 1, 256, False),
        ("pi05-encoder-padded", 456, 968, 456, 8, 1, 256, False),
    ]
    for name, sq, sk, valid_k, hq, hk, dim, causal in cases:
        q = torch.randn(1, sq, hq, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(1, sk, hk, dim, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        seqused_k = (
            torch.tensor([valid_k], device="cuda", dtype=torch.int32)
            if valid_k is not None else None
        )
        fa4_ms = timed_ms(
            lambda: ops.forward_static(
                q, k, v, out, causal=causal, seqused_k=seqused_k
            )
        )
        ref_k = k[:, :valid_k] if valid_k is not None else k
        ref_v = v[:, :valid_k] if valid_k is not None else v
        sdpa_ms = timed_ms(lambda: torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), ref_k.transpose(1, 2), ref_v.transpose(1, 2),
            is_causal=causal, enable_gqa=hq != hk))
        print(
            f"name={name} Sq={sq} Sk={sk} valid_k={valid_k} "
            f"H={hq}/{hk} D={dim} causal={causal} "
            f"fa4_ms={fa4_ms:.6f} sdpa_ms={sdpa_ms:.6f} "
            f"speedup={sdpa_ms/fa4_ms:.3f}x"
        )


if __name__ == "__main__":
    main()
