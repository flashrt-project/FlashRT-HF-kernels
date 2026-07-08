# flashrt-qkv-epilogue-train

Q/K/V projection + RoPE/split training API skeleton with eager autograd reference.

Hub repo: `flashrt/flashrt-qkv-epilogue-train`

## Public API

- `qkv_rope_reference`

## Status

This v1 package fixes the public training API, Hub packaging, autograd reference semantics, and acceptance harness. CUDA-optimized kernels are intentionally marked pending until they satisfy `kernel_acceptance_specs.md`.

No FP8/FP4 or low-precision math is used in this package.
