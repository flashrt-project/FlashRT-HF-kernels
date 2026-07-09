# flashrt-flex-attention-train

FlexAttention replacement API for PI-style prefix/action training masks.

## Available functions

- `flex_attention(q, k, v, prefix_len, action_block_size, ...)`
- `flex_attention_forward(...)`
- `reference_flex_attention(...)`
- `build_block_sparse_bool_masks(...)`

## Acceptance status

- Reference/SDPA autograd path: available.
- CUDA optimized implementation: pending A100/5090 acceptance.
- Precision mode: bf16/fp32 training reference, no FP8/FP4.
- Fallback: unsupported shapes route to SDPA.

Use this package to lock Lerobot/PI052 integration and run correctness and
benchmark gates before replacing the internal reference path with optimized
CUDA kernels.
