# flashrt-siglip-fwd-fusion

FP32 SigLIP no-grad forward fusion API skeleton with eager reference.

## Available functions

- `siglip_residual_layernorm_fwd`
- `siglip_gelu_fwd`
- `use_fused_siglip_path`

## Acceptance status

- Reference/eager autograd path: available.
- CUDA optimized implementation: pending acceptance.
- Precision mode: fp32/bf16 training reference only, no FP8/FP4.

Use this package to lock API integration and run correctness harnesses before replacing the internal reference path with optimized CUDA kernels.
