# FlashRT Spatiotemporal Layout

This package provides FlashRT spatiotemporal layout helpers for Hugging Face
Kernel Hub.

It is intended for VLA, video, diffusion, and world-model pipelines that move
between latent `(B, C, T, H, W)` tensors, token matrices, temporal unshuffle
layouts, and short latent caches.

## Kernels

- `ncdhw_to_blc_bf16`: convert NCDHW BF16 latents to BLC token matrices.
- `time_unshuffle2_bf16`: convert `(B, 2C, T, H, W)` to
  `(B, C, 2T, H, W)`.
- `add_bias_ncdhw_bf16`: in-place BF16 channel-bias add for NCDHW latents.
- `update_cache2_ncdhw_bf16`: maintain a two-frame NCDHW latent cache.

## Hardware

- CUDA 12.8+
- BF16-capable NVIDIA GPUs

Current local source validation is on RTX 5090. Broader hardware rows should be
added after installed-artifact validation.

## Upstream

The serving source of truth remains FlashRT:

https://github.com/LiangSu8899/FlashRT
