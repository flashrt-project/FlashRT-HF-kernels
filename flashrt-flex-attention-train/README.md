# flashrt-flex-attention-train

FlexAttention replacement training package for PI-style dual-expert
transformers.

Hub repo: `flashrt/flashrt-flex-attention-train`

## Public API

- `flex_attention`
- `flex_attention_forward`
- `reference_flex_attention`
- `build_block_sparse_bool_masks`
- `backend_marker`

## Scope

This package locks the public Tensor API and correctness harness for a native
replacement of the PI052 FlexAttention/SDPA attention path:

- prefix self-attention rows
- action-to-prefix rows plus block-diagonal action rows
- `head_dim=256`
- BF16 forward/backward through PyTorch autograd fallback
- detached-prefix semantics for action rows reading prefix K/V
- prefix mask, prefix padding mask, action block mask, and action padding mask
- automatic SDPA fallback for unsupported shapes

The current implementation is the SDPA-backed training reference. It is meant
to be the stable integration target for native CUDA fwd/bwd kernels; no native
performance claim is made until the benchmark gates in `VALIDATION.md` pass on
both A100 and RTX 5090.
