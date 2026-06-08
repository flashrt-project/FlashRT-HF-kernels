# FlashRT Residual Norm Quant

This package provides FlashRT BF16 RMSNorm and residual-add RMSNorm kernels with
static FP8 E4M3 output for Hugging Face Kernel Hub.

It is designed as a runtime glue package. Use it when a model pipeline needs to
feed the next FP8 GEMM/FFN block without doing residual add, RMSNorm, and FP8
quantization as separate PyTorch operations.

## Kernels

- `rms_norm_bf16`: BF16 RMSNorm with affine weight.
- `rms_norm_quant_fp8_static_bf16`: BF16 RMSNorm followed by static-scale FP8
  E4M3 quantization.
- `residual_add_rms_norm_quant_fp8_static_bf16`: in-place BF16 residual add,
  RMSNorm, and static-scale FP8 E4M3 quantization.

## Hardware

- CUDA 12.8+
- NVIDIA GPUs with FP8 E4M3 support

Current local source validation is on RTX 5090. Broader hardware rows should be
added after installed-artifact validation.

## Upstream

The serving source of truth remains FlashRT:

https://github.com/LiangSu8899/FlashRT
