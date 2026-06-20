#!/usr/bin/env python3
"""Minimal Hub usage for world-model-conv."""

from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    wmc = get_kernel("flashrt/world-model-conv")
    cache = torch.randn((1, 2, 16, 16, 32), device="cuda").to(torch.float8_e4m3fn)
    new = torch.randn((1, 4, 16, 16, 32), device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn((32, 3, 3, 3, 32), device="cuda").to(torch.float8_e4m3fn)
    bias = torch.zeros((32,), device="cuda", dtype=torch.bfloat16)
    residual = torch.zeros((1, 32, 4, 16, 16), device="cuda", dtype=torch.bfloat16)
    out = wmc.fp8_conv3d_v18_ncdhw_res_bf16out(cache, new, weight, bias, residual)
    print(out.shape)


if __name__ == "__main__":
    main()
