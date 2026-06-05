---
tags:
- flashrt
- kernel
- cuda
- vision
- video
- diffusion
- vla
---

# FlashRT VLA and Video Kernels

Reusable VLA, vision, video, and diffusion kernels from FlashRT.

The first implemented slice targets Q/K post-processing:

- `q_norm_rope_bf16`: per-head RMSNorm plus rotate-half RoPE for Q staging.
- `k_norm_rope_v_cache_bf16`: per-head RMSNorm plus rotate-half RoPE for K,
  plus V cache copy.
- `qkv_split_norm_rope_bf16`: packed QKV split plus Q/K RMSNorm and
  interleaved RoPE for video/VLA token blocks.

## When To Use

Use this package when a model produces packed BF16 QKV and then performs
separate QKV split, Q/K RMSNorm, and RoPE before attention. This pattern is
common in VLA, vision-language, video, and diffusion transformer blocks, and
is often missing from generic LLM kernel collections.

For fair model-block attribution, keep the same QKV projection and attention
implementation on both paths and replace only the postprocess island.

## Hardware

CUDA GPU with BF16 support. The current implementation is optimized for
head_dim=128 decode and video/VLA token paths.

See `examples/qkv_postprocess_block.py` for a minimal HF-style module using
`qkv_split_norm_rope_bf16`.
