#!/usr/bin/env python3
"""Benchmark linear-attention-primitives against PyTorch eager."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "tests"))
from test_linear_attention_primitives import load_installed_ops, load_source_ops  # noqa: E402


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
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(123)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    print("| Workload | Shape | FlashRT us | PyTorch eager us | Speedup |")
    print("|---|---:|---:|---:|---:|")

    for k, n in [(128, 512), (4096, 1024), (5120, 10240)]:
        x = (torch.randn((k,), device="cuda") * 0.05).to(torch.bfloat16)
        w = (torch.randn((n, k), device="cuda") * 0.05).to(torch.bfloat16)
        fused = bench(lambda: ops.bf16_matvec(x, w), args.warmup, args.iters)
        eager = bench(lambda: x @ w.t(), args.warmup, args.iters)
        print(f"| bf16_matvec | N={n},K={k} | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

    for m, k, n in [(2, 5120, 96), (3, 5120, 96), (4, 5120, 96)]:
        x = (torch.randn((m, k), device="cuda") * 0.05).to(torch.bfloat16)
        w = (torch.randn((n, k), device="cuda") * 0.05).to(torch.bfloat16)
        fused = bench(lambda: ops.bf16_smallm_matmul(x, w), args.warmup, args.iters)
        eager = bench(lambda: x @ w.t(), args.warmup, args.iters)
        print(f"| bf16_smallm_matmul | M={m},N={n},K={k} | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

    for rows in [1, 128, 1024]:
        packed = torch.randn((rows, (16 + 16 + 48) * 128), device="cuda", dtype=torch.bfloat16)
        fused = bench(lambda: ops.split_qkv_broadcast_bf16(packed, 16, 16, 48, 128), args.warmup, args.iters)
        eager = bench(
            lambda: (
                packed[:, : 16 * 128].reshape(rows, 16, 128)[:, torch.arange(48, device="cuda") * 16 // 48].contiguous(),
                packed[:, 16 * 128 : 32 * 128].reshape(rows, 16, 128)[:, torch.arange(48, device="cuda") * 16 // 48].contiguous(),
                packed[:, 32 * 128 :].reshape(rows, 48, 128).contiguous(),
            ),
            args.warmup,
            args.iters,
        )
        print(f"| split_qkv_broadcast | rows={rows},heads=16/48,dim=128 | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

    for rows in [1, 128, 1024]:
        q = torch.randn((rows, 16, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((rows, 16, 128), device="cuda", dtype=torch.bfloat16)
        cos = torch.randn((rows, 64), device="cuda", dtype=torch.bfloat16)
        sin = torch.randn((rows, 64), device="cuda", dtype=torch.bfloat16)

        def eager_rope():
            half = 32
            qo = q.clone()
            ko = k.clone()
            qo[:, :, :half] = ((-q[:, :, half:64].float() * sin[:, None, :half].float()).to(torch.bfloat16).float() + q[:, :, :half].float() * cos[:, None, :half].float()).to(torch.bfloat16)
            qo[:, :, half:64] = ((q[:, :, :half].float() * sin[:, None, half:64].float()).to(torch.bfloat16).float() + q[:, :, half:64].float() * cos[:, None, half:64].float()).to(torch.bfloat16)
            ko[:, :, :half] = ((-k[:, :, half:64].float() * sin[:, None, :half].float()).to(torch.bfloat16).float() + k[:, :, :half].float() * cos[:, None, :half].float()).to(torch.bfloat16)
            ko[:, :, half:64] = ((k[:, :, :half].float() * sin[:, None, half:64].float()).to(torch.bfloat16).float() + k[:, :, half:64].float() * cos[:, None, half:64].float()).to(torch.bfloat16)
            return qo, ko

        fused = bench(lambda: ops.partial_rope_qk_bf16(q, k, cos, sin, 64), args.warmup, args.iters)
        eager = bench(eager_rope, args.warmup, args.iters)
        print(f"| partial_rope_qk | rows={rows},heads=16,dim=128,rope=64 | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

    for rows in [1, 128, 1024]:
        a = torch.randn((rows, 64), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((rows, 64), device="cuda", dtype=torch.bfloat16)
        neg = torch.randn((48,), device="cuda", dtype=torch.float32) * 0.1
        bias = torch.randn((48,), device="cuda", dtype=torch.float32) * 0.1
        fused = bench(lambda: ops.gated_delta_prepare_bf16(a, b, neg, bias, heads=48, a_stride=64, b_stride=64), args.warmup, args.iters)
        eager = bench(
            lambda: (
                neg[None, :] * torch.nn.functional.softplus(a[:, :48].float() + bias[None, :])
            ).to(torch.bfloat16),
            args.warmup,
            args.iters,
        )
        print(f"| gated_delta_prepare | rows={rows},heads=48 | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
