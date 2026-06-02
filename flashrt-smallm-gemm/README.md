# flashrt-smallm-gemm

Draft package for FlashRT small-M GEMM and GEMV kernels.

This package should target decode and low-latency shapes without exposing
model-specific names.

## Scope

Initial candidate APIs:

- `smallm_fp8_gemm`
- `smallm_nvfp4_gemm`
- `splitk_decode_gemv`
- `smallm_bf16_matmul`

## Naming Rule

Public names should describe shapes and dtype behavior. Model names may appear
only in benchmark labels or provenance notes.

## Baselines

Benchmarks should compare against cuBLASLt, generic CUTLASS, and PyTorch eager
where appropriate.

## Showcase Criteria

- Focus on M=1, small batch, and split-K decode shapes where launch and latency
  dominate.
- Include wide-N and wide-K cases only when the tuned tile policy is competitive
  against cuBLASLt or a strong CUTLASS baseline.
- Document hardware-specific paths separately instead of presenting one kernel
  as universally supported.
