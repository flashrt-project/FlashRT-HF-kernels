#!/usr/bin/env python3
"""Minimal Hub example for static FP8 linear+bias projection."""

from __future__ import annotations

import argparse

import torch
from kernels import get_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    ops = get_kernel(
        "flashrt/flashrt-fp8-ffn", version=1, trust_remote_code=True
    )
    device = torch.device("cuda")
    fp8_dtype = torch.float8_e4m3fn
    m, k, n = 51, 1536, 4608

    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    weight = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.02
    bias = torch.randn((n,), device=device, dtype=torch.bfloat16) * 0.01
    input_scale = x.abs().amax().float().reshape(1).clamp_min(1e-6) / 448.0
    weight_scale = weight.abs().amax().float().reshape(1).clamp_min(1e-6) / 448.0
    weight_fp8 = torch.clamp(
        weight.float() / weight_scale, -448, 448
    ).to(fp8_dtype)

    input_fp8 = torch.empty((m, k), device=device, dtype=fp8_dtype)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    def project(value: torch.Tensor) -> torch.Tensor:
        return ops.bf16_fp8_linear_bias_bf16(
            value,
            weight_fp8,
            bias,
            input_scale,
            weight_scale,
            input_fp8=input_fp8,
            out=out,
        )

    call = torch.compile(project, fullgraph=True) if args.compile else project
    result = call(x)
    torch.cuda.synchronize()
    print(result.shape, result.dtype, torch.isfinite(result).all().item())


if __name__ == "__main__":
    main()
