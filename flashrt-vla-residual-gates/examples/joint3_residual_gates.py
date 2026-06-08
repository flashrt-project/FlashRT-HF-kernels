#!/usr/bin/env python3
"""Minimal HF Hub-style VLA residual/gate example."""

from __future__ import annotations

import argparse

import torch

try:
    from kernels import get_kernel
except ModuleNotFoundError:
    get_kernel = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="flashrt/flashrt-vla-residual-gates")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--video-rows", type=int, default=2520)
    parser.add_argument("--action-rows", type=int, default=16)
    parser.add_argument("--und-rows", type=int, default=16)
    parser.add_argument("--dim", type=int, default=3072)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    if get_kernel is not None:
        ops = get_kernel(args.repo_id, version=args.version, trust_remote_code=True)
    else:
        import flashrt_vla_residual_gates as ops

    def segment(rows: int):
        residual = torch.randn((rows, args.dim), device="cuda", dtype=torch.bfloat16)
        x = torch.randn_like(residual)
        gate = torch.randn_like(residual)
        return residual, x, gate

    v_residual, v_x, v_gate = segment(args.video_rows)
    a_residual, a_x, a_gate = segment(args.action_rows)
    u_residual = torch.randn((args.und_rows, args.dim), device="cuda", dtype=torch.bfloat16)
    u_x = torch.randn_like(u_residual)
    v_bias = torch.zeros((args.dim,), device="cuda", dtype=torch.bfloat16)

    fn = ops.joint3_bias_gate_residual_action_nobias_bf16
    if args.compile:
        fn = torch.compile(fn, fullgraph=True, mode="reduce-overhead")

    v_out, a_out, u_out = fn(
        v_residual,
        v_x,
        v_bias,
        v_gate,
        a_residual,
        a_x,
        a_gate,
        u_residual,
        u_x,
    )
    torch.cuda.synchronize()
    mode = "torch.compile(fullgraph=True)" if args.compile else "eager"
    print(
        f"{mode} v={tuple(v_out.shape)} a={tuple(a_out.shape)} "
        f"u={tuple(u_out.shape)} dtype={v_out.dtype}"
    )


if __name__ == "__main__":
    main()
