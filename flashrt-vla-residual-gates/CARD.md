# FlashRT VLA Residual Gates

This package provides FlashRT joint residual/gate kernels for Hugging Face
Kernel Hub.

It is intended for VLA/video model blocks that maintain separate video, action,
and und token groups but update them in the same block.

## Kernels

- `joint3_bias_gate_residual_bf16`: fused video/action/und residual updates
  where video and action both use bias and gate.
- `joint3_bias_gate_residual_action_nobias_bf16`: fused video/action/und
  residual updates where video uses bias+gate, action uses gate, and und uses a
  plain residual add.

## Hardware

- CUDA 12.8+
- BF16-capable NVIDIA GPUs

Current local source validation is on RTX 5090. Broader hardware rows should be
added after installed-artifact validation.

## Upstream

The serving source of truth remains FlashRT:

https://github.com/LiangSu8899/FlashRT
