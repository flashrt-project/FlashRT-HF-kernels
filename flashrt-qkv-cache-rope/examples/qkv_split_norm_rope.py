#!/usr/bin/env python3
"""Minimal HF Hub-style QKV split/RMSNorm/RoPE example."""

from __future__ import annotations

import argparse

import torch

try:
    from kernels import get_kernel
except ModuleNotFoundError:
    get_kernel = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="flashrt/flashrt-qkv-cache-rope")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--joint3", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    if get_kernel is not None:
        ops = get_kernel(args.repo_id, version=args.version, trust_remote_code=True)
    else:
        import flashrt_qkv_cache_rope as ops

    dim = args.heads * args.head_dim
    if args.joint3:
        video_len = args.seq_len
        action_len = 16
        und_len = 16
        packed_v = torch.randn((1, video_len, 3 * dim), device="cuda", dtype=torch.bfloat16)
        packed_a = torch.randn((1, action_len, 3 * dim), device="cuda", dtype=torch.bfloat16)
        packed_u = torch.randn((1, und_len, 3 * dim), device="cuda", dtype=torch.bfloat16)
        qkv_v_bias = torch.zeros((3 * dim,), device="cuda", dtype=torch.bfloat16)
        q_w = torch.ones((dim,), device="cuda", dtype=torch.bfloat16)
        k_w = torch.ones((dim,), device="cuda", dtype=torch.bfloat16)
        theta = torch.randn((video_len, args.head_dim // 2), device="cuda")
        freqs_re = torch.cos(theta).contiguous()
        freqs_im = torch.sin(theta).contiguous()
        q_cat = torch.empty((1, video_len + action_len + und_len, args.heads, args.head_dim), device="cuda", dtype=torch.bfloat16)
        k_cat = torch.empty_like(q_cat)
        v_cat = torch.empty_like(q_cat)
        fn = ops.qkv_split_joint3_cat_bf16
        if args.compile:
            fn = torch.compile(fn, fullgraph=True, mode="reduce-overhead")
        q, k, v = fn(
            packed_v,
            qkv_v_bias,
            q_w,
            k_w,
            freqs_re,
            freqs_im,
            packed_a,
            q_w,
            k_w,
            packed_u,
            q_w,
            k_w,
            heads=args.heads,
            head_dim=args.head_dim,
            q_cat_out=q_cat,
            k_cat_out=k_cat,
            v_cat_out=v_cat,
        )
        torch.cuda.synchronize()
        mode = "torch.compile(fullgraph=True)" if args.compile else "eager"
        print(
            f"{mode} joint3 q shape={tuple(q.shape)} k shape={tuple(k.shape)} "
            f"v shape={tuple(v.shape)} dtype={q.dtype}"
        )
        return

    packed_qkv = torch.randn(
        (args.batch, args.seq_len, 3 * dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_w = torch.ones((dim,), device="cuda", dtype=torch.bfloat16)
    k_w = torch.ones((dim,), device="cuda", dtype=torch.bfloat16)
    theta = torch.randn((args.seq_len, args.head_dim // 2), device="cuda")
    freqs_re = torch.cos(theta).contiguous()
    freqs_im = torch.sin(theta).contiguous()

    fn = ops.qkv_split_norm_rope_bf16
    if args.compile:
        fn = torch.compile(fn, fullgraph=True, mode="reduce-overhead")

    q, k = fn(
        packed_qkv,
        q_w,
        k_w,
        freqs_re,
        freqs_im,
        heads=args.heads,
        head_dim=args.head_dim,
    )
    torch.cuda.synchronize()
    mode = "torch.compile(fullgraph=True)" if args.compile else "eager"
    print(f"{mode} q shape={tuple(q.shape)} k shape={tuple(k.shape)} dtype={q.dtype}")


if __name__ == "__main__":
    main()
