from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    ops = get_kernel("flashrt/int8-transformer-primitives", version=1)
    x = torch.randn((8, 1024), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((2560, 1024), device="cuda", dtype=torch.bfloat16)
    x_i8, x_scale = ops.quantize_int8_rowwise_bf16(x)
    w_i8, w_scale = ops.quantize_int8_rowwise_bf16(w)
    y = ops.int8_rowwise_linear_bf16(x_i8, w_i8, x_scale, w_scale)
    torch.cuda.synchronize()
    print(y.shape, y.dtype)


if __name__ == "__main__":
    main()
