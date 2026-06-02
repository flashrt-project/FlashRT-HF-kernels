---
tags:
- flashrt
- kernel
- cuda
- gemm
- fp8
- nvfp4
---

# FlashRT GEMM and FP8 Quant Epilogues

Fused GEMM and FP8 quantization epilogue kernels from FlashRT.

The main current surface is the post-GEMM FP8 quantization epilogue slice:
BF16 input plus optional BF16 bias, GELU(tanh), and FP8 e4m3 quantized output,
plus a per-channel BF16 scaling and FP8 quantization primitive.

The package also exposes BF16 GEMM wrappers using cuBLASLt fused bias and GELU
epilogues. These wrappers are shape-sensitive and should be evaluated for the
target workload before promotion.

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

The v1 performance message for this package should center on the FP8
quantization helpers. GEMM epilogue numbers should be reported per shape, not
as a broad claim.

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

See `examples/fp8_quant_epilogue_block.py` for a minimal HF-style module using
the FP8 quantization epilogue helpers.
