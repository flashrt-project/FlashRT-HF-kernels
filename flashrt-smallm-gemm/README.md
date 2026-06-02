# flashrt-smallm-gemm

Draft package for FlashRT small-M GEMM and GEMV kernels.

This package should target decode and low-latency shapes without exposing
model-specific names.

## Scope

Implemented draft API:

- `nvfp4_w4a4_decode_matvec_bf16out`

Selected next APIs:

- `nvfp4_w4a4_smallm_warpsplit_bf16out`
- `tiny_fp8_smallm_gemm_bf16out`

These are intentionally shape-specialized. The package should be honest about
supported M/K/N families and should expose dispatch wrappers only after each
tile policy has a benchmark-backed reason to exist.

The first synced source slice is an SM120 NVFP4 W4A4 M=1 decode matvec with
BF16 output. It currently supports `K in {4096, 12288}` and expects packed FP4
activation/weight bytes plus CUTLASS Sm1xx swizzled UE4M3 scale-factor buffers.
The package has local source-extension correctness and a public HF-style
benchmark harness. It remains draft until benchmark results, fair baselines, and
full HF builder validation are recorded.

## Naming Rule

Public names should describe shapes and dtype behavior. Model names may appear
only in benchmark labels or provenance notes.

## Baselines

Benchmarks should compare against cuBLASLt, generic CUTLASS, and PyTorch eager
where appropriate.

## Validation

The first draft source slice compiles as a local source extension and passes a
deterministic correctness smoke for `K=4096` and `K=12288` on RTX 5090. See
`VALIDATION.md` for the exact scope and remaining gaps.

## Example

`examples/nvfp4_w4a4_decode_matvec.py` shows deterministic M=1 decode matvec
usage with packed FP4 bytes and swizzled scale-factor buffers.

## Showcase Criteria

- Focus on M=1, small batch, and split-K decode shapes where launch and latency
  dominate.
- Include wide-N and wide-K cases only when the tuned tile policy is competitive
  against cuBLASLt or a strong CUTLASS baseline.
- Document hardware-specific paths separately instead of presenting one kernel
  as universally supported.

See `SELECTED_KERNELS.md` for the first concrete source slices.
