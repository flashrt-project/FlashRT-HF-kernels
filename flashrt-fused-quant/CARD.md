---
tags:
- kernel
- cuda
- quantization
- rmsnorm
- swiglu
---

# FlashRT Fused Quantization

Draft kernel card for FlashRT memory-bound fused activation, normalization,
residual, and low-bit quantization kernels.

This package is not ready for Hub publication yet.

## Selected First Features

- Split and merged `SiLU(gate) * up` plus NVFP4 swizzled quantization. This
  first draft wrapper is synced and has Tensor bindings.
- RMSNorm plus FP4/SFA quantization.
- Residual update plus RMSNorm plus FP4/SFA quantization.
- BF16-safe residual/RMSNorm variant with generic public naming.

## Status

This package stays draft until the activation+quant path has public benchmark
grids, memory-bandwidth results, and full HF builder validation. Tensor
bindings and fake-quant byte-parity tests exist for the first source slice.
