# flashrt-adarms-train

AdaRMS + gated residual training API skeleton with eager autograd reference.

Hub repo: `flashrt/flashrt-adarms-train`

## Public API

- `adarms_forward`
- `resgate_adarms_forward`
- `FlashRTAdaRMSNorm`

## Status

This v1 package fixes the public training API, Hub packaging, autograd reference semantics, and acceptance harness. CUDA-optimized kernels are intentionally marked pending until they satisfy `kernel_acceptance_specs.md`.

No FP8/FP4 or low-precision math is used in this package.
