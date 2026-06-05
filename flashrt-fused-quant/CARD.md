---
tags:
- flashrt
- kernel
- cuda
- quantization
- rmsnorm
- swiglu
---

# FlashRT Fused Quantization

FlashRT memory-bound fused activation and low-bit quantization kernels.

The v1 surface focuses on SwiGLU-style activation products followed by NVFP4
swizzled quantization. These kernels are useful when the model already has
BF16 `gate` and `up` tensors and needs to materialize packed NVFP4 activations
plus CUTLASS Sm1xx-compatible scale-factor buffers.

## Kernels

- `silu_mul_quant_nvfp4_swizzled_bf16`: split `gate` and `up` tensors,
  compute `SiLU(gate) * up`, pack to NVFP4, and emit swizzled UE4M3 scales.
- `silu_mul_merged_quant_nvfp4_swizzled_bf16`: same operation for merged
  `[gate | up]` input layout.
- `nvfp4_swizzled_scale_bytes`: helper for output scale-buffer sizing.

## When To Use

Use this package for memory-bound activation-plus-quantization islands in
SwiGLU/MLP blocks, especially when the next consumer expects NVFP4 packed data
and Sm1xx swizzled scale-factor layout.

Do not compare this package against a CPU/Python reference as a headline
speedup. Public benchmark rows should report GPU latency and a fair GPU
baseline where available.

See the repository usage guide and the package example:
https://github.com/LiangSu8899/FlashRT-HF-kernels/blob/main/docs/usage.md
https://github.com/LiangSu8899/FlashRT-HF-kernels/blob/main/flashrt-fused-quant/examples/swiglu_nvfp4_quant_block.py

## Hardware

Current v1 build scope is CUDA 12.8+ SM120.
