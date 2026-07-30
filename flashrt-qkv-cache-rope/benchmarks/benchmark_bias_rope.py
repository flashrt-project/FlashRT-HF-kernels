#!/usr/bin/env python3
"""Benchmark fused packed-QKV bias, split, and rotate-half RoPE."""

from __future__ import annotations

import argparse
import torch

from benchmark import load_installed_ops, load_source_ops, metrics, time_us


SHAPES = [
    ("groot_vit", 277, 16, 16, 64, torch.bfloat16),
    ("qwen3_vl_vision", 1024, 16, 16, 72, torch.bfloat16),
    ("lingbot_vision", 1024, 16, 16, 80, torch.bfloat16),
    ("lingbot_attention_fp16", 51, 16, 8, 80, torch.float16),
    ("qwen3_vl_text", 277, 32, 8, 128, torch.bfloat16),
    ("wan_video", 2520, 24, 24, 128, torch.bfloat16),
]


def reference(
    packed, bias, cos, sin, q_heads, kv_heads, head_dim, output_dtype
):
    batch, seq_len, _ = packed.shape
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    biased = packed.float() + bias.float().view(1, 1, -1)
    q = biased[..., :q_dim].view(batch, seq_len, q_heads, head_dim)
    k = biased[..., q_dim : q_dim + kv_dim].view(
        batch, seq_len, kv_heads, head_dim
    )
    v = biased[..., q_dim + kv_dim :].to(output_dtype).view(
        batch, seq_len, kv_heads, head_dim
    )
    half = head_dim // 2
    c = cos[..., :half].unsqueeze(2)
    s = sin[..., :half].unsqueeze(2)

    def rotate(x):
        out = torch.empty_like(x)
        out[..., :half] = x[..., :half] * c - x[..., half:] * s
        out[..., half:] = x[..., half:] * c + x[..., :half] * s
        return out.to(output_dtype)

    return rotate(q), rotate(k), v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(47)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    print("| shape | B,S,QH,KVH,HD | output | wrapper us | eager us | compile us | vs eager | vs compile | p99 | cosine |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, rows, q_heads, kv_heads, head_dim, output_dtype in SHAPES:
        width = (q_heads + 2 * kv_heads) * head_dim
        packed = torch.randn((1, rows, width), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn((width,), device="cuda", dtype=torch.bfloat16)
        theta = torch.randn(
            (1, rows, head_dim // 2), device="cuda", dtype=torch.float32
        )
        cos, sin = theta.cos().contiguous(), theta.sin().contiguous()
        q_out = torch.empty(
            (1, rows, q_heads, head_dim), device="cuda", dtype=output_dtype
        )
        k_out = torch.empty(
            (1, rows, kv_heads, head_dim), device="cuda", dtype=output_dtype
        )
        v_out = torch.empty_like(k_out)

        def wrapper_call():
            op = (
                ops.qkv_split_bias_rope_bf16
                if output_dtype == torch.bfloat16
                else ops.qkv_split_bias_rope_fp16
            )
            return op(
                packed, bias, cos, sin, q_heads, kv_heads, head_dim,
                q_out, k_out, v_out,
            )

        def eager_call():
            return reference(
                packed, bias, cos, sin, q_heads, kv_heads, head_dim,
                output_dtype
            )

        compiled_call = torch.compile(eager_call, fullgraph=True)
        expected = eager_call()
        compiled_call()
        got = wrapper_call()
        torch.cuda.synchronize()
        p99_values, cosine_values = [], []
        for actual, ref in zip(got, expected):
            p99, cosine = metrics(actual, ref)
            p99_values.append(p99)
            cosine_values.append(cosine)
        wrapper_us = time_us(wrapper_call, args.warmup, args.iters)
        eager_us = time_us(eager_call, args.warmup, args.iters)
        compile_us = time_us(compiled_call, args.warmup, args.iters)
        print(
            f"| {name} | 1,{rows},{q_heads},{kv_heads},{head_dim} | "
            f"{str(output_dtype).removeprefix('torch.')} | "
            f"{wrapper_us:.3f} | {eager_us:.3f} | {compile_us:.3f} | "
            f"{eager_us / wrapper_us:.2f}x | {compile_us / wrapper_us:.2f}x | "
            f"{max(p99_values):.6f} | {min(cosine_values):.8f} |"
        )


if __name__ == "__main__":
    main()
