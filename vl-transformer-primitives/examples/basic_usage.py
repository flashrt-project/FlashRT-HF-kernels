#!/usr/bin/env python3
"""Minimal Hub usage for flashrt/vl-transformer-primitives."""

from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    vl = get_kernel("flashrt/vl-transformer-primitives")

    q_pre = torch.randn((32, 128), device="cuda", dtype=torch.bfloat16)
    q_norm_weight = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
    theta = torch.randn((64,), device="cuda", dtype=torch.float32)
    cos = torch.cos(theta).to(torch.bfloat16)
    sin = torch.sin(theta).to(torch.bfloat16)
    q_out = vl.qwen3_q_norm_rope_qstage_bf16(q_pre, q_norm_weight, cos, sin)

    tokens = torch.randn((2 * 16 * 16, 1152), device="cuda", dtype=torch.bfloat16)
    pooled = vl.avg_pool_vision_tokens_bf16(tokens, nv=2, h=16, w=16, pool_factor=2)

    torch.cuda.synchronize()
    print(q_out.shape, pooled.shape)


if __name__ == "__main__":
    main()
