# FlashRT Adaptive Norms

This package provides FlashRT adaptive normalization kernels for Hugging Face
Kernel Hub.

It is intended for DiT, VLA, video, and world-model blocks that use RMSNorm
plus per-row style scale/shift/gate parameters.

## Kernels

- `ada_rms_norm_style_bf16`: RMSNorm + style scale/shift, with BF16 output and
  BF16 gate extraction.
- `gate_residual_ada_norm_fp8_static_bf16`: in-place gated residual update,
  AdaRMSNorm, static FP8 E4M3 output, and BF16 gate extraction.

## Hardware

- CUDA 12.8+
- BF16-capable NVIDIA GPUs
- FP8-capable NVIDIA GPUs for the FP8 output API

Current local source validation is on RTX 5090. Broader hardware rows should be
added after installed-artifact validation.

## Upstream

The serving source of truth remains FlashRT:

https://github.com/LiangSu8899/FlashRT
