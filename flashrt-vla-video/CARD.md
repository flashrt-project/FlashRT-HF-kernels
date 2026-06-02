---
tags:
- kernel
- cuda
- vision
- video
- diffusion
- vla
---

# FlashRT VLA and Video Kernels

Reusable VLA, vision, video, and diffusion kernels from FlashRT.

The first implemented slice targets decode-time Q/K post-processing:

- `q_norm_rope_bf16`: per-head RMSNorm plus rotate-half RoPE for Q staging.
- `k_norm_rope_v_cache_bf16`: per-head RMSNorm plus rotate-half RoPE for K,
  plus V cache copy.

## Planned Features

- Generic benchmark tables for Q/K decode post-processing.
- Patch embedding data movement and bias/position fusion.
- Video and 3D convolution low-bit helper kernels.
- DiT/VAE-style normalization and quantization helpers.

## Hardware

CUDA GPU with BF16 support. The current implementation is optimized for
head_dim=128 decode paths.
