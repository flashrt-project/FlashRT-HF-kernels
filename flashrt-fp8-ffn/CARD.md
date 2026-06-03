# FlashRT FP8 FFN

This package provides Hugging Face Kernel Hub wrappers for FlashRT FP8 FFN
building blocks.

## Kernels

- `fp8_gemm_bf16`
- `fp8_linear_bias_gelu_quant_bf16`
- `fp8_gelu_mlp_bf16`

## Hardware

- CUDA 12.8+
- FP8-capable NVIDIA GPUs with cuBLASLt FP8 support

Current local validation is on RTX 5090. Other hardware should be added to the
benchmark matrix before broader claims.

## Notes

This package is a Tensor API integration layer. The upstream serving source of
truth remains FlashRT. Shape-locked SM120 megakernels are intentionally not
included in this generic package.
