#!/usr/bin/env python3
"""Benchmark diffusion-step-ops against PyTorch eager references."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "tests"))
from test_diffusion_step_ops import load_installed_ops, load_source_ops  # noqa: E402


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
    torch.manual_seed(1234)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    print("| Workload | Shape | FlashRT us | PyTorch eager us | Speedup |")
    print("|---|---:|---:|---:|---:|")

    for shape in [(1024,), (16384,), (2, 16, 32, 64), (1, 16, 17, 64, 64)]:
        a = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        fused = bench(lambda: ops.add_bf16(a, b), args.warmup, args.iters)
        eager = bench(lambda: (a.float() + b.float()).to(torch.bfloat16), args.warmup, args.iters)
        print(f"| add_bf16 | {tuple(shape)} | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

        fused = bench(lambda: ops.euler_step_bf16(a, b, -0.125), args.warmup, args.iters)
        eager = bench(lambda: (a.float() + b.float() * -0.125).to(torch.bfloat16), args.warmup, args.iters)
        print(f"| euler_step_bf16 | {tuple(shape)} | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

        residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        residual_ref = residual.clone()
        fused = bench(lambda: ops.cfg_combine_into_residual_bf16(residual, a, b, 4.5), args.warmup, args.iters)
        eager = bench(lambda: residual_ref.add_((b.float() + 4.5 * (a.float() - b.float())).to(torch.bfloat16)), args.warmup, args.iters)
        print(f"| cfg_combine_bf16 | {tuple(shape)} | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

    for shape in [(1, 4, 5, 16, 16), (2, 8, 9, 32, 32), (1, 16, 17, 64, 64)]:
        video = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        cond = torch.randn((shape[0], shape[1], shape[3], shape[4]), device="cuda", dtype=torch.bfloat16)
        video_ref = video.clone()
        fused = bench(lambda: ops.teacher_force_first_frame_bf16(video, cond), args.warmup, args.iters)
        eager = bench(lambda: video_ref[:, :, 0].copy_(cond), args.warmup, args.iters)
        print(f"| teacher_force_first_frame | {tuple(shape)} | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

        fused = bench(lambda: ops.motus_decode_postprocess_bf16_to_fp32(video), args.warmup, args.iters)
        eager = bench(lambda: ((video[:, :, 1:].float() + 1.0) * 0.5).clamp(0.0, 1.0).contiguous(), args.warmup, args.iters)
        print(f"| decode_postprocess | {tuple(shape)} | {fused:.3f} | {eager:.3f} | {eager / fused:.2f}x |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
