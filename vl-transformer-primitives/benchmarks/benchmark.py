#!/usr/bin/env python3
"""Benchmark vl-transformer-primitives against PyTorch eager references."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PACKAGE = Path(__file__).resolve().parents[1]
TESTS = PACKAGE / "tests"
sys.path.insert(0, str(TESTS))
from test_vl_transformer_primitives import (  # noqa: E402
    load_installed_ops,
    load_source_ops,
    make_decode_case,
    ref_avg_pool,
    ref_norm_rope,
)


def bench(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) * 1000.0 / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(1234)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    print("| Workload | Shape | FlashRT us | PyTorch eager us | Speedup |")
    print("|---|---:|---:|---:|---:|")

    for heads in [1, 4, 8, 16, 32, 40]:
        q, k, v, q_w, k_w, cos, sin = make_decode_case(heads)
        fused = bench(
            lambda: ops.qwen3_q_norm_rope_qstage_bf16(q, q_w, cos, sin),
            args.warmup,
            args.iters,
        )
        eager = bench(
            lambda: ref_norm_rope(q, q_w, cos, sin, 1e-6),
            args.warmup,
            args.iters,
        )
        print(f"| q_norm_rope | heads={heads}, d=128 | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

        fused = bench(
            lambda: ops.qwen3_k_norm_rope_kvwrite_bf16(k, v, k_w, cos, sin),
            args.warmup,
            args.iters,
        )
        eager = bench(
            lambda: (ref_norm_rope(k, k_w, cos, sin, 1e-6), v.clone()),
            args.warmup,
            args.iters,
        )
        print(f"| k_norm_rope_vwrite | heads={heads}, d=128 | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

    for nv, h, w, dim, pool in [
        (1, 16, 16, 1024, 2),
        (2, 16, 16, 1152, 2),
        (4, 16, 16, 2048, 4),
        (2, 32, 32, 1024, 4),
    ]:
        x = torch.randn((nv * h * w, dim), device="cuda", dtype=torch.bfloat16)
        fused = bench(
            lambda: ops.avg_pool_vision_tokens_bf16(x, nv, h, w, pool),
            args.warmup,
            args.iters,
        )
        eager = bench(
            lambda: ref_avg_pool(x, nv, h, w, pool),
            args.warmup,
            args.iters,
        )
        print(
            f"| avg_pool_vision | nv={nv}, h={h}, w={w}, dim={dim}, pool={pool} "
            f"| {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
