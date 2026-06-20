---
license: apache-2.0
tags:
- cuda
- flashrt
- diffusers
- video
- world-model
- fp8
- blackwell
---

# FlashRT World Model Conv

Native CUDA FP8 3D convolution kernels for world-model and video diffusion
runtime hot paths.

## Available Functions

- `fp8_conv3d_v18_ncdhw_res_bf16out`

## Usage

```python
from kernels import get_kernel

wmc = get_kernel("flashrt/world-model-conv")
out = wmc.fp8_conv3d_v18_ncdhw_res_bf16out(
    cache_x_fp8,
    new_x_fp8,
    weight_fp8,
    bias_bf16,
    residual_bf16,
    alpha=0.75,
)
```

The function takes NDHWC FP8 cache/new inputs and writes BF16 NCDHW output.
It is intended for static-buffer diffusion/world-model runtimes where avoiding
`torch.cat`, output slicing, and separate bias/residual launches matters.

## Hardware

This kernel uses Blackwell architecture-specific FP8 MMA instructions and is
compiled for `cuda-capabilities = ["12.0a"]`.
