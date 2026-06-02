# flashrt-nvfp4

Package for FlashRT NVFP4 primitives, layout conversion helpers, and selected
low-bit GEMM surfaces.

This package should expose data format and layout operations that other kernel
packages can depend on or mirror.

## Scope

Current APIs:

- `nvfp4_sf_linear_to_swizzled`
- `nvfp4_sf_swizzled_bytes`

Planned APIs:

- `nvfp4_linear_bias_gelu_fp4out_sm120`
- `nvfp4_linear_bias_gelu_bf16out_sm120`
- `nvfp4_linear_streamk_bias_bf16out_sm120`

The first source slice is the scale-factor layout helper, because it is small,
reusable, and makes the fused GEMM outputs inspectable. The GEMM epilogue
surfaces follow once CUTLASS include requirements and Tensor binding
constraints are isolated.

## Non-Goals

- Do not expose full model-specific inference paths.
- Do not hide architecture constraints. SM120/SM120a-only kernels must be
  labeled as such.

## Baselines

Correctness should compare against PyTorch dequant references and FlashRT
internal outputs.

Performance should compare against a strong CUTLASS/cuBLAS path when applicable,
not only against PyTorch eager.

## Validation

See `VALIDATION.md` for the current tested shapes, RTX 5090 smoke environment,
and remaining builder/build gaps.

## Showcase Criteria

- Make NVFP4/SFA/SFB layout constraints explicit.
- Separate data-movement helpers from GEMM APIs in docs and benchmarks.
- Treat Blackwell-only speedups as a strong but architecture-scoped story.

See `SELECTED_KERNELS.md` for the exact FlashRT source provenance and promotion
order.
