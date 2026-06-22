from __future__ import annotations

import torch
from kernels import get_kernel


def main() -> None:
    ops = get_kernel("flashrt/adaptive-layernorm-producers", version=1, trust_remote_code=True)

    rows, dim = 2520, 3072
    x = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    scale = torch.zeros((dim,), device="cuda", dtype=torch.bfloat16)
    shift = torch.zeros((dim,), device="cuda", dtype=torch.bfloat16)
    act_scale = torch.tensor([0.025], device="cuda", dtype=torch.float32)

    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ops.ada_layer_norm_quant_fp8_bf16(x, scale, shift, act_scale, out=out)

    packed, sf = ops.ada_layer_norm_quant_nvfp4_swizzled_bf16(x, scale, shift)
    print(out.shape, out.dtype)
    print(packed.shape, packed.dtype, sf.numel())


if __name__ == "__main__":
    main()
