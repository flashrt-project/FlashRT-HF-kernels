# flashrt-adarms-train

AdaRMS + gated residual training kernels for BF16/FP32 training.

Hub repo: `flashrt/flashrt-adarms-train`

## Public API

- `adarms`
- `resgate_adarms`
- `adarms_forward`
- `resgate_adarms_forward`
- `FlashRTAdaRMSNorm`

## Status

This package includes CUDA forward/backward kernels for the supported CUDA
BF16/FP32 shapes and falls back to the PyTorch reference path for CPU,
unsupported dtypes, unsupported hidden sizes, and gradcheck-only FP64 inputs.

Correctness Gate 1 passes on the local RTX 5090 validation matrix. Performance
Gate 2 is 11/12: all adaptive conditioning paths pass; the remaining
conditional case is large-row non-adaptive gated residual RMSNorm, which is not
the target pi052 adaptive path and is documented in `benchmarks/RESULTS.md`.

No FP8/FP4 or low-precision math is used in this package.
