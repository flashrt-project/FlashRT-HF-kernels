# flashrt-fused-quant

Draft package for FlashRT non-GEMM fused quantization kernels.

This package should expose memory-bound fused kernels that are broadly useful
in Transformer, VLA, and diffusion forward passes.

## Scope

Implemented draft APIs:

- `silu_mul_quant_nvfp4_swizzled_bf16`
- `silu_mul_merged_quant_nvfp4_swizzled_bf16`

Selected next APIs:

- `residual_rmsnorm_quant_nvfp4_swizzled_bf16`
- `rmsnorm_quant_nvfp4_sfa_fp16`
- `residual_rmsnorm_quant_nvfp4_sfa_fp16`

## Non-Goals

- Do not include GEMM epilogues; those belong in `flashrt-gemm-epilogues`.
- Do not include NVFP4 layout-only helpers; those belong in `flashrt-nvfp4`.
- Do not expose model-specific public names.

## Baselines

Benchmarks should compare against PyTorch eager sequences and FlashRT internal
reference output where available.

## Validation

The first source slice is synced and has Tensor bindings and package-local
fake-quant reference tests. It has passed local source-extension compile and
split/merged byte-parity smoke on RTX 5090. The package remains draft until the
benchmark grids, bandwidth results, and full HF builder validation are recorded.

## Example

`examples/swiglu_nvfp4_quant_block.py` shows split and merged FFN gate/up usage
with generic public API names.

See `SELECTED_KERNELS.md` for the first concrete source slices and why they are
kept separate from GEMM epilogues.
