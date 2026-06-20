from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    ops = get_kernel("flashrt/causal-conv1d-state", version=1, trust_remote_code=True)
    b, s, c, k = 1, 8, 10240, 4
    x = torch.randn(b, s, c, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(c, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(c, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(b, c, k - 1, device="cuda", dtype=torch.bfloat16)
    out = ops.causal_conv1d_update_chunk_parallel_bf16(x, w, state, bias)
    print(out.shape, state.shape)


if __name__ == "__main__":
    main()
