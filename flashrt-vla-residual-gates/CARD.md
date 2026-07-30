# FlashRT VLA Residual Gates

This package provides FlashRT joint residual/gate kernels for Hugging Face
Kernel Hub.

It is intended for VLA/video model blocks that maintain separate video, action,
and und token groups but update them in the same block.

## Kernels

- `bias_residual_bf16`: fused `residual + x + bias` for one BF16 token
  segment.
- `joint3_bias_gate_residual_bf16`: fused video/action/und residual updates
  where video and action both use bias and gate.
- `joint3_bias_gate_residual_action_nobias_bf16`: fused video/action/und
  residual updates where video uses bias+gate, action uses gate, and und uses a
  plain residual add.
- `joint3_bias_fp8_gate_residual_bf16`: the same joint update with a
  statically scaled FP8 E4M3 video gate.
- `joint3_bias_fp8_gate_residual_action_nobias_bf16`: FP8 video gate with no
  action bias.

## Hardware

- CUDA 12.8+
- BF16-capable NVIDIA GPUs

Current local source validation is on RTX 5090. Broader hardware rows should be
added after installed-artifact validation.

## Upstream

The serving source of truth remains FlashRT:

https://github.com/LiangSu8899/FlashRT
