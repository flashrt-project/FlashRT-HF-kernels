# flashrt-adarms-train

AdaRMS + gated residual training API skeleton with eager autograd reference.

## Available functions

- `adarms_forward`
- `resgate_adarms_forward`
- `FlashRTAdaRMSNorm`

## Acceptance status

- Reference/eager autograd path: available.
- CUDA optimized implementation: pending acceptance.
- Precision mode: fp32/bf16 training reference only, no FP8/FP4.

Use this package to lock API integration and run correctness harnesses before replacing the internal reference path with optimized CUDA kernels.
