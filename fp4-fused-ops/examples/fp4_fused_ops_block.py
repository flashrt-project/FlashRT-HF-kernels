#!/usr/bin/env python3
"""Minimal Hub-style call for flashrt/fp4-fused-ops."""

from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    ops = get_kernel("flashrt/fp4-fused-ops", version=1, trust_remote_code=True)

    merged = torch.randn((16, 4096), device="cuda", dtype=torch.float16)
    packed, sfa = ops.silu_mul_fp4_sfa_v2_fp16(merged)
    dequant = ops.dequantize_fp4_sfa_fp16(packed, sfa)

    print("packed", tuple(packed.shape), packed.dtype)
    print("sfa", tuple(sfa.shape), sfa.dtype)
    print("debug_dequant", tuple(dequant.shape), dequant.dtype)


if __name__ == "__main__":
    main()
