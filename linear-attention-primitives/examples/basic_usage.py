#!/usr/bin/env python3
"""Minimal Hub usage for linear-attention-primitives."""

from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    lap = get_kernel("flashrt/linear-attention-primitives")
    x = torch.randn((4, 5120), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((96, 5120), device="cuda", dtype=torch.bfloat16)
    out = lap.bf16_smallm_matmul(x, w)
    print(out.shape)


if __name__ == "__main__":
    main()
