# flashrt/adaptive-layernorm-producers

FlashRT native CUDA adaptive LayerNorm producer kernels for DiT, Wan-style
video diffusion, and VLA/runtime demo pipelines.

This package fuses normalization/modulation and low-precision activation
production before FP8 or NVFP4 GEMM consumers.

## Functions

- `ada_layer_norm_quant_fp8_bf16`
- `ada_layer_norm_quant_fp8_modfp8_bf16`
- `awq_ada_layer_norm_quant_fp8_bf16`
- `ada_layer_norm_quant_nvfp4_swizzled_bf16`
- `ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16`
- `layer_norm_no_affine_quant_fp8_static_bf16`
- `swizzled_sf_size`

## Quick Start

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/adaptive-layernorm-producers", version=1, trust_remote_code=True)

x = torch.randn((2520, 3072), device="cuda", dtype=torch.bfloat16)
scale = torch.zeros((3072,), device="cuda", dtype=torch.bfloat16)
shift = torch.zeros((3072,), device="cuda", dtype=torch.bfloat16)
act_scale = torch.tensor([0.025], device="cuda", dtype=torch.float32)

x_fp8 = ops.ada_layer_norm_quant_fp8_bf16(x, scale, shift, act_scale)
```

Use `README.md` for tensor contracts and `VALIDATION.md` for correctness and
benchmark status.
