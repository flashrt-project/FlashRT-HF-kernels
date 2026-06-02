---
tags:
- kernel
- cuda
- nvfp4
- quantization
---

# FlashRT NVFP4

Kernel card for FlashRT NVFP4 layout helpers and planned fused Blackwell
low-bit GEMM epilogues.

The first buildable slice exposes NVFP4 scale-factor layout conversion.

## Features

- `nvfp4_sf_linear_to_swizzled` and `nvfp4_sf_swizzled_bytes`.

## Planned Features

- SM120/SM120a NVFP4 GEMM with fused bias+GELU and BF16 output.
- SM120/SM120a NVFP4 GEMM with fused bias+GELU and FP4 output quantization.
- Stream-K down-projection GEMM with optional bias.

## Status

The fused GEMM epilogues follow after CUTLASS dependency isolation.

Current validation status is recorded in `VALIDATION.md`.
