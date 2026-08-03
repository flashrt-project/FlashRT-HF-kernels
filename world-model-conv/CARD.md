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

- `bf16_causal_conv3d_ndhwc_bf16` (SM110 experimental, no speedup claim)
- `fp8_conv3d_v18_ncdhw_res_bf16out`
- `fp8_causal_conv3d_ndhwc_bf16`
- `fp8_conv2d_3x3_nhwc_bf16`
- `fp8_conv2d_3x3_ncdhw_bf16`
- `nvfp4_causal_conv3d_ndhwc_bf16`
- `nvfp4_causal_conv3d_residual_ncdhw_bf16`

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
compiled for SM120a. The separate BF16 probe artifact is compiled for SM110a;
it is exposed for development and parity work but does not replace cuDNN.

On SM110a the FP8/NVFP4 ops use a portable pure-SIMT reference
(`portable_conv_simt.cu`), since no native tensor-core backend exists for them
on Thor; the BF16 conv3d keeps its native SM110 kernel. SM120 always uses the
tensor-core kernels. Set `FLASHRT_FORCE_SIMT=1` to route any device through
the SIMT reference (used by the correctness test to validate parity against
the native path).
