from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    attn = get_kernel("flashrt/fp8-kv-attention", trust_remote_code=True)
    q = torch.randn(1, 24, 256, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(8, 128, 4, 256, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    v_cache = torch.randn(8, 128, 4, 256, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    out = attn.xqa_bf16_fp8kv(q, k_cache, v_cache)
    torch.cuda.synchronize()
    print(out.shape, out.dtype)


if __name__ == "__main__":
    main()
