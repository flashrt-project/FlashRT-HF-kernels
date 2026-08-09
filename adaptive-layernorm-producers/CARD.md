# flashrt/adaptive-layernorm-producers

FlashRT native CUDA adaptive LayerNorm producer kernels for DiT, Wan-style
video diffusion, and VLA/runtime demo pipelines.

This package fuses normalization/modulation and low-precision activation
production before FP8 or NVFP4 GEMM consumers.

## Functions

- `ada_layer_norm_quant_fp8_bf16`
- `ada_layer_norm_quant_fp8_ptok_bf16`
- `ada_layer_norm_quant_fp8_ptok_table_bf16`
- `ada_layer_norm_quant_fp8_modfp8_bf16`
- `awq_ada_layer_norm_quant_fp8_bf16`
- `ada_layer_norm_quant_nvfp4_swizzled_bf16`
- `ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16`
- `layer_norm_no_affine_quant_fp8_static_bf16`
- `layer_norm_no_affine_quant_nvfp4_swizzled_bf16`
- `adaln_modulation6_bf16`
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

Per-token modulation and table-fused modulation are available for video DiT
and VLA blocks whose shift/scale values vary by token:

```python
ptok_scale = torch.zeros_like(x)
ptok_shift = torch.zeros_like(x)
x_fp8 = ops.ada_layer_norm_quant_fp8_ptok_bf16(
    x, ptok_scale, ptok_shift, act_scale
)

chunks = 6
temb = torch.zeros((x.shape[0], chunks, x.shape[1]), device="cuda",
                   dtype=torch.bfloat16)
table = torch.zeros((chunks, x.shape[1]), device="cuda",
                    dtype=torch.float32)
x_fp8 = ops.ada_layer_norm_quant_fp8_ptok_table_bf16(
    x, temb, table, act_scale, shift_idx=0, scale_idx=1
)
```

The `torch211-cxx11-cu130-aarch64-linux` variant is a native fat binary with
SM87 (Jetson AGX Orin) and SM110a (Jetson AGX Thor) code objects. The SM110a
path is runtime-validated on Thor; SM87 runtime validation remains pending on
Orin hardware. Runtime variant selection is handled by `get_kernel`.

Use `README.md` for tensor contracts and `VALIDATION.md` for correctness and
benchmark status.
