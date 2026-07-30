# Kernel card: padded-fp8-producers

## Purpose

Remove separate normalization, activation, quantization, and row-padding
launches before fixed-tile FP8 GEMMs.

## Numerical contract

- BF16 normalization and modulation seam before FP8 conversion.
- E4M3FN saturation range `[-448, 448]`.
- FP16 merged SwiGLU input still rounds the activation product through BF16
  before static FP8 quantization, matching the production model seam.
- Padded rows are exactly zero.
- Residual input is not mutated; the updated BF16 residual is returned through
  `residual_out`.

## Functions

- `adaptive_rms_norm_quant_fp8_padded_bf16`
- `residual_add_adaptive_rms_norm_quant_fp8_padded_bf16`
- `swiglu_quant_fp8_padded_bf16`
- `swiglu_merged_quant_fp8_padded_bf16`
- `swiglu_merged_quant_fp8_padded_fp16`

## Scope

CUDA 12.8 or newer. This package contains CUDA kernels and does not claim ROCm
support. Shape and architecture claims are limited to tested build artifacts.
