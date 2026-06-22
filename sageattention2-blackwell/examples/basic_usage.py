from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    ops = get_kernel("flashrt/sageattention2-blackwell", version=1, trust_remote_code=True)

    q = torch.randn((1, 1024, 32, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1024, 8, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((1, 1024, 8, 128), device="cuda", dtype=torch.bfloat16)

    out = ops.sage2_prefill_f16_bf16_d128(q, k, v, causal=True)
    torch.cuda.synchronize()
    print(out.shape, out.dtype)


if __name__ == "__main__":
    main()
