# flashrt-rope-train

RoPE fwd/bwd training API skeleton with eager autograd reference.

## Available functions

- `apply_rope_train`
- `apply_rope_backward_reference`

## Acceptance status

- Reference/eager autograd path: available.
- CUDA optimized implementation: pending acceptance.
- Precision mode: fp32/bf16 training reference only, no FP8/FP4.

Use this package to lock API integration and run correctness harnesses before replacing the internal reference path with optimized CUDA kernels.
