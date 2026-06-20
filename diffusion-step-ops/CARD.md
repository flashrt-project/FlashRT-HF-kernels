---
license: apache-2.0
tags:
  - kernels
  - cuda
  - flashrt
  - diffusers
  - diffusion
  - cuda-graph
---

# flashrt/diffusion-step-ops

FlashRT CUDA kernels for diffusion step glue that is commonly left in PyTorch
eager code.

## Functions

```python
from kernels import get_kernel

ops = get_kernel("flashrt/diffusion-step-ops")

ops.add_bf16(a, b)
ops.euler_step_bf16(latent, velocity, dt=-0.125)
ops.cfg_combine_into_residual_bf16(residual, v_cond, v_uncond, beta=4.5)
ops.cfg_combine_into_residual_fp16(residual_fp16, v_cond_fp16, v_uncond_fp16, beta=4.5)
ops.teacher_force_first_frame_bf16(video_latent, cond_latent)
ops.motus_decode_postprocess_bf16_to_fp32(decoded)
ops.cast_bf16_to_fp32(src)
```

This package is meant for static-buffer / CUDA Graph diffusion runtimes and
Diffusers demos. It is not a scheduler replacement by itself; it provides the
low-level tensor operations used inside optimized scheduler/runtime loops.
