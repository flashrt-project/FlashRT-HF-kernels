#!/usr/bin/env python3
"""Benchmark transformer layout primitives."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "transformer-layout-primitives" / "tests"))
from test_transformer_layout_primitives import (  # noqa: E402
    load_source_ops,
    qk_pair_rmsnorm_rope_ref,
    qk_rmsnorm_rope_ref,
    rotate_half_ref,
)


def _apply_mem_cap(max_mem_gb: float = 30.0) -> None:
    if not torch.cuda.is_available() or max_mem_gb <= 0:
        return
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(max_mem_gb * 1024**3)
    if total <= 0 or cap >= total:
        return
    torch.cuda.set_per_process_memory_fraction(cap / total)


def load_ops(backend: str, artifact: str | None):
    if backend == "source":
        return load_source_ops()
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("transformer_layout_primitives")
    finally:
        if artifact:
            sys.path.remove(artifact)


def time_us(fn, warmup: int, iters: int) -> float:
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
    parser.add_argument("--mode", choices=["headline", "full"], default="headline")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--max-mem-gb", type=float, default=30.0)
    args = parser.parse_args()
    _apply_mem_cap(args.max_mem_gb)
    ops = load_ops(args.backend, args.artifact)

    print("workload,shape,op,flashrt_us,torch_eager_us,speedup")

    repeat_shapes = [("gqa_prefill", 2520, 8, 128, 4), ("decode_gqa", 1, 8, 128, 4)]
    if args.mode == "full":
        repeat_shapes += [("short_prefill", 128, 8, 128, 4), ("vl_prefill", 4096, 8, 128, 4)]
    for name, seq, heads, dim, repeat in repeat_shapes:
        src = torch.randn((seq, heads, dim), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((seq, heads * repeat, dim), device="cuda", dtype=torch.bfloat16)

        def flash():
            ops.repeat_interleave_heads_bf16(src, repeat, out=out)

        def eager():
            src.repeat_interleave(repeat, dim=1)

        fu = time_us(flash, args.warmup, args.iters)
        eu = time_us(eager, max(5, args.warmup // 2), max(20, args.iters // 2))
        print(f"{name},{seq}x{heads}x{dim}x{repeat},repeat_interleave_heads_bf16,{fu:.3f},{eu:.3f},{eu/fu:.2f}x")

    rope_shapes = [("qwen_prefill", 4096, 32, 128), ("video_prefill", 2520, 24, 128)]
    if args.mode == "full":
        rope_shapes += [("decode", 1, 32, 128), ("short_prefill", 128, 32, 128)]
    for name, seq, heads, dim in rope_shapes:
        x = torch.randn((seq, heads, dim), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((dim,), device="cuda", dtype=torch.bfloat16)
        cos = torch.randn((seq, dim), device="cuda", dtype=torch.bfloat16)
        sin = torch.randn((seq, dim), device="cuda", dtype=torch.bfloat16)
        out = x.clone()

        def flash():
            out.copy_(x)
            ops.qk_rmsnorm_rope_bf16_(out, weight, cos, sin)

        def eager():
            qk_rmsnorm_rope_ref(x, weight, cos, sin)

        fu = time_us(flash, args.warmup, args.iters)
        eu = time_us(eager, max(5, args.warmup // 2), max(20, args.iters // 2))
        print(f"{name},{seq}x{heads}x{dim},qk_rmsnorm_rope_bf16_,{fu:.3f},{eu:.3f},{eu/fu:.2f}x")

        def flash_rope():
            out.copy_(x)
            ops.rope_rotate_half_bf16_(out, cos, sin)

        def eager_rope():
            rotate_half_ref(x, cos, sin)

        fu = time_us(flash_rope, args.warmup, args.iters)
        eu = time_us(eager_rope, max(5, args.warmup // 2), max(20, args.iters // 2))
        print(f"{name},{seq}x{heads}x{dim},rope_rotate_half_bf16_,{fu:.3f},{eu:.3f},{eu/fu:.2f}x")

    pair_shapes = [
        ("groot_n17_llm", 277, 16, 8, 128),
        ("qwen3vl_vision", 1024, 16, 16, 72),
        ("lingbot_vision", 1024, 16, 16, 80),
        ("video_transformer", 2520, 24, 24, 128),
    ]
    if args.mode == "full":
        pair_shapes += [
            ("action_boundary", 51, 16, 16, 64),
            ("wan_partial_tile", 5070, 24, 24, 128),
        ]
    for name, rows, q_heads, k_heads, dim in pair_shapes:
        q = torch.randn((rows, q_heads, dim), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((rows, k_heads, dim), device="cuda", dtype=torch.bfloat16)
        q_weight = torch.randn((dim,), device="cuda", dtype=torch.bfloat16)
        k_weight = torch.randn((dim,), device="cuda", dtype=torch.bfloat16)
        cos = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
        sin = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        def flash_pair():
            ops.qk_pair_rmsnorm_rope_bf16(
                q, k, q_weight, k_weight, cos, sin, q_out=q_out, k_out=k_out
            )

        def eager_pair():
            qk_pair_rmsnorm_rope_ref(
                q, k, q_weight, k_weight, cos, sin
            )

        fu = time_us(flash_pair, args.warmup, args.iters)
        eu = time_us(eager_pair, max(5, args.warmup // 2), max(20, args.iters // 2))
        print(
            f"{name},{rows}x{q_heads}+{k_heads}x{dim},"
            f"qk_pair_rmsnorm_rope_bf16,{fu:.3f},{eu:.3f},{eu/fu:.2f}x"
        )

    batch, seq, dim = 8, 2048, 2048
    x = torch.randn((batch * seq, dim), device="cuda", dtype=torch.bfloat16)
    gathered = torch.empty((2 * batch, dim), device="cuda", dtype=torch.bfloat16)

    def flash_gather():
        ops.text_gather_bf16(x, batch, seq, out=gathered)

    def eager_gather():
        torch.stack([x[b * seq + offset] for b in range(batch) for offset in (0, seq - 1)], dim=0)

    fu = time_us(flash_gather, args.warmup, args.iters)
    eu = time_us(eager_gather, max(5, args.warmup // 2), max(20, args.iters // 2))
    print(f"text_tokens,{batch}x{seq}x{dim},text_gather_bf16,{fu:.3f},{eu:.3f},{eu/fu:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
