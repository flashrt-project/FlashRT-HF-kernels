# flashrt-siglip-fwd-fusion

FP32 SigLIP no-grad forward fusion API skeleton with eager reference.

Hub repo: `flashrt/flashrt-siglip-fwd-fusion`

## Public API

- `siglip_residual_layernorm_fwd`
- `siglip_gelu_fwd`
- `use_fused_siglip_path`

## Status

This v1 package fixes the public training API, Hub packaging, autograd reference semantics, and acceptance harness. CUDA-optimized kernels are intentionally marked pending until they satisfy `kernel_acceptance_specs.md`.

No FP8/FP4 or low-precision math is used in this package.
