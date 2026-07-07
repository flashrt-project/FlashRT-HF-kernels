from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    ops = get_kernel("flashrt/transformer-layout-primitives", version=1)
    q = torch.randn((128, 32, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
    cos = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    sin = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    ops.qk_rmsnorm_rope_bf16_(q, weight, cos, sin)
    torch.cuda.synchronize()
    print(q.shape, q.dtype)


if __name__ == "__main__":
    main()
