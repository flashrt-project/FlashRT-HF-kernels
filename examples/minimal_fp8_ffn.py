#!/usr/bin/env python3
"""Minimal FlashRT FP8 FFN call through Hugging Face Kernel Hub."""

from __future__ import annotations

import torch
from kernels import get_kernel


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    ops = get_kernel("flashrt/flashrt-fp8-ffn", version=1, trust_remote_code=True)

    m, k, h, n = 128, 1024, 4096, 1024
    x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    up_w_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)
    hidden_scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    down_w_scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)

    x_fp8 = quantize_fp8(
        torch.randn((m, k), device="cuda", dtype=torch.bfloat16),
        x_scale,
    )
    up_w_fp8 = quantize_fp8(
        torch.randn((h, k), device="cuda", dtype=torch.bfloat16),
        up_w_scale,
    )
    down_w_fp8 = quantize_fp8(
        torch.randn((n, h), device="cuda", dtype=torch.bfloat16),
        down_w_scale,
    )
    up_bias = torch.randn((h,), device="cuda", dtype=torch.bfloat16)
    down_bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)

    y = ops.fp8_gelu_mlp_bf16(
        x_fp8,
        up_w_fp8,
        up_bias,
        down_w_fp8,
        down_bias,
        x_scale,
        up_w_scale,
        hidden_scale,
        down_w_scale,
    )
    torch.cuda.synchronize()
    print(f"output shape={tuple(y.shape)} dtype={y.dtype}")


if __name__ == "__main__":
    main()
