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

## Performance Notes

FP8 quantize epilogue helpers are the strongest current surface across the
local shape suite. BF16 GEMM epilogue wrappers are shape-sensitive and should be
evaluated against `torch.addmm`/`gelu(torch.addmm)` for target shapes before
promotion.

## Hardware

CUDA GPUs with BF16 tensor core support are expected for the GEMM path. FP8
output helpers additionally require PyTorch and hardware/runtime support for
`torch.float8_e4m3fn`.

## Validation

Validated HF builder targets currently include
`torch211-cxx11-cu128-x86_64-linux` and
`torch211-cxx11-cu126-x86_64-linux`, and
`torch211-cxx11-cu130-x86_64-linux`. Builder ABI checks passed for all three
variants. Host-side correctness smoke passed on an RTX 5090 with PyTorch
2.9.1+cu128. See `VALIDATION.md` for the full record and remaining gaps.
