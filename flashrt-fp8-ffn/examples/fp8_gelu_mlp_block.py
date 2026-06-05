#!/usr/bin/env python3
"""Minimal HF-style FP8 GELU MLP block example."""

from __future__ import annotations

import argparse

import torch

try:
    from kernels import get_kernel
except ModuleNotFoundError:
    get_kernel = None


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="flashrt/flashrt-fp8-ffn")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--h", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    if get_kernel is not None:
        ops = get_kernel(args.repo_id, version=args.version, trust_remote_code=True)
    else:
        import flashrt_fp8_ffn as ops

    x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    up_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    hidden_scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    down_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)

    x = quantize_fp8(torch.randn((args.m, args.k), device="cuda", dtype=torch.bfloat16), x_scale)
    up_w = quantize_fp8(torch.randn((args.h, args.k), device="cuda", dtype=torch.bfloat16), up_scale)
    down_w = quantize_fp8(torch.randn((args.n, args.h), device="cuda", dtype=torch.bfloat16), down_scale)
    up_b = torch.randn((args.h,), device="cuda", dtype=torch.bfloat16)
    down_b = torch.randn((args.n,), device="cuda", dtype=torch.bfloat16)

    fn = ops.fp8_gelu_mlp_bf16
    if args.compile:
        fn = torch.compile(fn, fullgraph=True, mode="reduce-overhead")

    y = fn(x, up_w, up_b, down_w, down_b, x_scale, up_scale, hidden_scale, down_scale)
    torch.cuda.synchronize()
    mode = "torch.compile(fullgraph=True)" if args.compile else "eager"
    print(f"{mode} fp8_gelu_mlp_bf16 output shape={tuple(y.shape)} dtype={y.dtype}")


if __name__ == "__main__":
    main()
