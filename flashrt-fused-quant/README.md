# flashrt-fused-quant

Draft package for FlashRT non-GEMM fused quantization kernels.

This package should expose memory-bound fused kernels that are broadly useful
in Transformer, VLA, and diffusion forward passes.

## Scope

Initial candidate APIs:

- `rmsnorm_quant`
- `residual_rmsnorm_quant`
- `layernorm_quant`
- `swiglu_quant`
- `geglu_quant`
- `qkv_split_rope`
- `qkv_split_rope_kvwrite`

## Non-Goals

- Do not include GEMM epilogues; those belong in `flashrt-gemm-epilogues`.
- Do not include NVFP4 layout-only helpers; those belong in `flashrt-nvfp4`.
- Do not expose model-specific public names.

## Baselines

Benchmarks should compare against PyTorch eager sequences and FlashRT internal
reference output where available.
