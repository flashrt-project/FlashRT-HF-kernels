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

- Split and merged `SiLU(gate) * up` plus NVFP4 swizzled quantization.
- RMSNorm plus FP4/SFA quantization.
- Residual update plus RMSNorm plus FP4/SFA quantization.
- BF16-safe residual/RMSNorm variant with generic public naming.

## Status

This package stays draft until the activation+quant path has Tensor bindings,
fake-quant reference tests, and memory-bandwidth benchmarks.
