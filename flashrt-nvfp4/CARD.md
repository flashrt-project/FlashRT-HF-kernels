---
tags:
- kernel
- cuda
- nvfp4
- quantization
---

# FlashRT NVFP4

Draft kernel card for FlashRT NVFP4 layout helpers and fused Blackwell
low-bit GEMM epilogues.

This package is not ready for Hub publication yet.

## Selected First Features

- `nvfp4_sf_linear_to_swizzled` and `nvfp4_sf_swizzled_bytes`.
- SM120/SM120a NVFP4 GEMM with fused bias+GELU and BF16 output.
- SM120/SM120a NVFP4 GEMM with fused bias+GELU and FP4 output quantization.
- Stream-K down-projection GEMM with optional bias.

## Status

This package stays draft until the layout helper has Tensor bindings, tests,
and package-local benchmarks. The fused GEMM epilogues follow after CUTLASS
dependency isolation.
