# flashrt-vocab-ce-train

Streaming huge-vocab CE training API skeleton with eager autograd reference.

Hub repo: `flashrt/flashrt-vocab-ce-train`

## Public API

- `vocab_ce_loss`
- `vocab_ce_fwd`

## Status

This v1 package fixes the public training API, Hub packaging, autograd reference semantics, and acceptance harness. CUDA-optimized kernels are intentionally marked pending until they satisfy `kernel_acceptance_specs.md`.

No FP8/FP4 or low-precision math is used in this package.
