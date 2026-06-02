---
tags:
- kernel
- cuda
- gemm
- decode
- fp8
- nvfp4
---

# FlashRT Small-M GEMM

Draft kernel card for FlashRT decode-oriented small-M GEMM/GEMV kernels.

This package is not ready for Hub publication yet.

## Selected First Features

- NVFP4 W4A4 decode matvec with BF16 output.
- NVFP4 W4A4 small-M warpsplit MMA with BF16 output.
- Tiny FP8 fixed-family small-M GEMM kernels.

## Status

This package stays draft until the shape-specialized APIs have explicit
supported shape grids, correctness tests, and latency benchmarks against
cuBLASLt/CUTLASS where applicable.
