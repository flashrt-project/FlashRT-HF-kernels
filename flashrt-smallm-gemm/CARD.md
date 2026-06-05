---
tags:
- flashrt
- kernel
- cuda
- gemm
- decode
- fp8
- nvfp4
---

# FlashRT Small-M GEMM

FlashRT decode-oriented small-M GEMM/GEMV kernels.

The v1 surface is a shape-specialized SM120 NVFP4 W4A4 M=1 matvec with BF16
output. It is meant for low-batch decode/projection paths where launch latency
and small-M efficiency dominate.

## Kernels

- `nvfp4_w4a4_decode_matvec_bf16out`: packed activation row and packed
  row-major weight matrix to BF16 output. Current supported K values are
  `{4096, 12288}`.

## When To Use

Use this package for Blackwell decode shapes after activations and weights
have already been packed to NVFP4/W4A4 layout and scale-factor buffers are in
CUTLASS Sm1xx swizzled format.

Unsupported K values should be treated as unsupported shapes, not slow fallback
paths. Callers should gate shapes before dispatching this kernel.

## Hardware

Current v1 build scope is CUDA 12.8+ SM120.
