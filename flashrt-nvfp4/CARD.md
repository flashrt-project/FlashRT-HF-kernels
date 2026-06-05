---
tags:
- flashrt
- kernel
- cuda
- nvfp4
- quantization
---

# FlashRT NVFP4

FlashRT NVFP4 layout helpers for Blackwell low-bit paths.

The v1 surface exposes NVFP4 scale-factor layout conversion. It converts
linear per-block scale bytes into the CUTLASS Sm1xx swizzled scale-factor
layout expected by downstream NVFP4 kernels.

## Kernels

- `nvfp4_sf_linear_to_swizzled`: convert `(rows, D / 16)` linear scale bytes
  to flat swizzled layout.
- `nvfp4_sf_swizzled_bytes`: compute the required swizzled output byte count.

## When To Use

Use this package as the layout bridge before calling Blackwell NVFP4 GEMM,
decode, or fused-quant kernels that consume Sm1xx swizzled scale factors.

For fused `SiLU(gate) * up` plus NVFP4 quantization, use
`flashrt/flashrt-fused-quant`. For W4A4 decode matvec, use
`flashrt/flashrt-smallm-gemm`.

## Hardware

Current validation status is recorded in `VALIDATION.md`.

Current v1 build scope is CUDA 12.8+ SM120.

See `examples/nvfp4_scale_factor_layout.py` for a minimal layout-conversion
example.
