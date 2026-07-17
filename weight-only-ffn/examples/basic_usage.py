#!/usr/bin/env python3
"""Minimal Kernel Hub usage for W4A16 and W8A16 FFN regions."""

import torch
from kernels import get_kernel


def main() -> None:
    ops = get_kernel(
        "flashrt/weight-only-ffn",
        version=1,
        trust_remote_code=True,
    )
    # Smallest documented VLA-shaped gated region accepted by both W4A16 and
    # W8A16 production auto dispatch.
    m, k, h, n = 1, 1024, 4096, 1024
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    gate_up = torch.randn((2 * h, k), device="cuda", dtype=torch.bfloat16)
    down = torch.randn((n, h), device="cuda", dtype=torch.bfloat16)

    for bits in (4, 8):
        quantize = getattr(ops, f"quantize_w{bits}_weight_bf16")
        ffn = getattr(ops, f"w{bits}a16_swiglu_ffn_bf16")
        gate_up_q, gate_up_scale = quantize(gate_up)
        down_q, down_scale = quantize(down)
        gate_up_tmp = torch.empty((m, 2 * h), device="cuda", dtype=torch.bfloat16)
        hidden_tmp = torch.empty((m, h), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        ffn(
            x,
            gate_up_q,
            gate_up_scale,
            down_q,
            down_scale,
            workspace=(gate_up_tmp, hidden_tmp),
            out=out,
        )
        torch.cuda.synchronize()
        print(f"W{bits}A16", tuple(out.shape), out.dtype)


if __name__ == "__main__":
    main()
