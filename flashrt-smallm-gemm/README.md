# flashrt-smallm-gemm

Draft package for FlashRT small-M GEMM and GEMV kernels.

This package should target decode and low-latency shapes without exposing
model-specific names.

## Scope

Selected first APIs:

- `nvfp4_w4a4_decode_matvec_bf16out`
- `nvfp4_w4a4_smallm_warpsplit_bf16out`
- `tiny_fp8_smallm_gemm_bf16out`

These are intentionally shape-specialized. The package should be honest about
supported M/K/N families and should expose dispatch wrappers only after each
tile policy has a benchmark-backed reason to exist.

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

See `SELECTED_KERNELS.md` for the first concrete source slices.
