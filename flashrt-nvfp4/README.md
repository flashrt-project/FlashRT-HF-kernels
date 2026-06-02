# flashrt-nvfp4

Draft package for FlashRT NVFP4 primitives, layout conversion helpers, and
selected low-bit GEMM surfaces.

This package should expose data format and layout operations that other kernel
packages can depend on or mirror.

## Scope

Initial candidate APIs:

- `quantize_nvfp4`
- `dequantize_nvfp4`
- `quantize_nvfp4_sfa`
- `nvfp4_linear`
- `nvfp4_linear_bias_gelu`
- `reshape_linear_scales_to_sfa`
- `sfa_size_bytes`

## Non-Goals

- Do not expose full model-specific inference paths.
- Do not hide architecture constraints. SM120/SM120a-only kernels must be
  labeled as such.

## Baselines

Correctness should compare against PyTorch dequant references and FlashRT
internal outputs.

Performance should compare against a strong CUTLASS/cuBLAS path when applicable,
not only against PyTorch eager.

## Showcase Criteria

- Make NVFP4/SFA/SFB layout constraints explicit.
- Separate data-movement helpers from GEMM APIs in docs and benchmarks.
- Treat Blackwell-only speedups as a strong but architecture-scoped story.
