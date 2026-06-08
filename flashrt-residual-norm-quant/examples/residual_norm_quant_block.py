#!/usr/bin/env python3
"""Minimal HF Hub-style residual/RMSNorm/static-FP8 quant example."""

from __future__ import annotations

import argparse

import torch

try:
    from kernels import get_kernel
except ModuleNotFoundError:
    get_kernel = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="flashrt/flashrt-residual-norm-quant")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    if get_kernel is not None:
        ops = get_kernel(args.repo_id, version=args.version, trust_remote_code=True)
    else:
        import flashrt_residual_norm_quant as ops

    x = torch.randn((args.rows, args.dim), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.ones((args.dim,), device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)

    fn = ops.residual_add_rms_norm_quant_fp8_static_bf16
    if args.compile:
        fn = torch.compile(fn, fullgraph=True, mode="reduce-overhead")

    out = fn(residual, x, weight, scale, eps=1e-6)
    torch.cuda.synchronize()
    mode = "torch.compile(fullgraph=True)" if args.compile else "eager"
    print(f"{mode} output shape={tuple(out.shape)} dtype={out.dtype}")


if __name__ == "__main__":
    main()
