---
tags:
- kernel
- cuda
- gemm
- fp8
- nvfp4
---

# FlashRT GEMM Epilogues

Fused GEMM epilogue kernels from FlashRT.

This package currently exposes a BF16 GEMM with cuBLASLt fused bias and GELU
epilogue. It also exposes the first post-GEMM epilogue slice:
BF16 input plus optional BF16 bias, GELU(tanh), and FP8 e4m3 quantized output,
plus a per-channel BF16 scaling and FP8 quantization primitive.

## Planned Features

- FP8 GEMM with fused bias and activation epilogues.
- NVFP4 GEMM with fused bias and activation epilogues.
- Quantized output epilogues for low-latency inference pipelines.
- Generic APIs for Transformer, VLA, and diffusion model linear blocks.

## Current API

- `bf16_gemm_bias(a, b, bias, out=None)`
- `bf16_gemm_bias_gelu(a, b, bias, out=None)`
- `bias_gelu_quantize_fp8_static_bf16(input, bias, scale, out=None)`
- `channel_scale_quantize_fp8_static_bf16(input, channel_scale, scale, out=None)`
- `gelu_quantize_fp8_static_bf16(input, scale, out=None)`

## Hardware

CUDA GPUs with BF16 tensor core support are expected for the GEMM path. FP8
output helpers additionally require PyTorch and hardware/runtime support for
`torch.float8_e4m3fn`.
