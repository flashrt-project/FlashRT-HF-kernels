#!/usr/bin/env python3
"""Minimal Hub-style call for flashrt/fp4-gemm."""

from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    ops = get_kernel("flashrt/fp4-gemm", version=1, trust_remote_code=True)

    x = torch.randn((32, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((512, 256), device="cuda", dtype=torch.float16)

    a_packed, sfa = ops.quantize_fp4_sfa_fp16(x, is_sfb=False)
    b_packed, sfb = ops.quantize_fp4_sfa_fp16(w, is_sfb=True)
    y = ops.fp4_w4a16_linear_bf16(a_packed, b_packed, sfa, sfb, alpha=1.0)

    print("a_packed", tuple(a_packed.shape), a_packed.dtype)
    print("b_packed", tuple(b_packed.shape), b_packed.dtype)
    print("output", tuple(y.shape), y.dtype)


if __name__ == "__main__":
    main()
