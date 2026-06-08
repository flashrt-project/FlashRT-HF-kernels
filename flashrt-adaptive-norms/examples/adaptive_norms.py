#!/usr/bin/env python3
"""Minimal HF Hub-style adaptive norms example."""

from __future__ import annotations

import argparse

import torch

try:
    from kernels import get_kernel
except ModuleNotFoundError:
    get_kernel = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="flashrt/flashrt-adaptive-norms")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--rows", type=int, default=2520)
    parser.add_argument("--dim", type=int, default=3072)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    if get_kernel is not None:
        ops = get_kernel(args.repo_id, version=args.version, trust_remote_code=True)
    else:
        import flashrt_adaptive_norms as ops

    x = torch.randn((args.rows, args.dim), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    gate = torch.randn_like(x)
    weight = torch.ones((args.dim,), device="cuda", dtype=torch.bfloat16)
    style = torch.randn((args.rows, 3 * args.dim), device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)

    fn = ops.gate_residual_ada_norm_fp8_static_bf16
    if args.compile:
        fn = torch.compile(fn, fullgraph=True, mode="reduce-overhead")

    residual, out_fp8, gate_out = fn(residual, x, gate, weight, style, scale)
    torch.cuda.synchronize()
    mode = "torch.compile(fullgraph=True)" if args.compile else "eager"
    print(
        f"{mode} residual={tuple(residual.shape)} out={tuple(out_fp8.shape)} "
        f"gate={tuple(gate_out.shape)} dtype={out_fp8.dtype}"
    )


if __name__ == "__main__":
    main()
