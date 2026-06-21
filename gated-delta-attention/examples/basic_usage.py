from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    ops = get_kernel("flashrt/gated-delta-attention", version=3, trust_remote_code=True)
    b, h, d = 1, 48, 128
    q = torch.randn(b, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn(b, h, device="cuda", dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(b, h, device="cuda")).to(torch.bfloat16)
    state = torch.zeros(b, h, d, d, device="cuda", dtype=torch.bfloat16)
    out = ops.gated_delta_recurrent_bf16(q, k, v, g, beta, state)
    print(out.shape)


if __name__ == "__main__":
    main()
