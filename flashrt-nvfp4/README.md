# flashrt-nvfp4

Draft package for FlashRT NVFP4 primitives and layout conversion helpers.

This package should expose data format and layout operations that other kernel
packages can depend on or mirror.

## Scope

Initial candidate APIs:

- `quantize_nvfp4`
- `dequantize_nvfp4`
- `quantize_nvfp4_sfa`
- `reshape_linear_scales_to_sfa`
- `sfa_size_bytes`

## Non-Goals

- Do not expose full model-specific inference paths.
- Do not make GEMM epilogue APIs the primary surface; those belong in
  `flashrt-gemm-epilogues`.

## Baselines

Correctness should compare against PyTorch dequant references and FlashRT
internal outputs.
