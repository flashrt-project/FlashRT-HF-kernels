#!/usr/bin/env python3
"""Minimal Hub usage for flashrt/diffusion-step-ops."""

from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    ops = get_kernel("flashrt/diffusion-step-ops")

    latent = torch.randn((1, 16, 17, 64, 64), device="cuda", dtype=torch.bfloat16)
    velocity = torch.randn_like(latent)
    updated = ops.euler_step_bf16(latent, velocity, dt=-0.125)

    cond = torch.randn((1, 16, 64, 64), device="cuda", dtype=torch.bfloat16)
    ops.teacher_force_first_frame_bf16(updated, cond)

    torch.cuda.synchronize()
    print(updated.shape)


if __name__ == "__main__":
    main()
