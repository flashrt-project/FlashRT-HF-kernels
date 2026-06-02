# Benchmark Results: flashrt-smallm-gemm

This file is the public result ledger for the v1 small-M GEMM block. It is
currently a pre-release template plus local validation status, not a final
release table.

Validated on June 2, 2026.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- Hardware scope: CUDA 12.8+ SM120 local validation only so far
- Benchmark path: pending built package artifact

## Current Scope

| API | Scope | Current status |
| --- | --- | --- |
| `nvfp4_w4a4_decode_matvec_bf16out` | SM120 NVFP4 W4A4 M=1 decode matvec with BF16 output | Source synced, Tensor binding present, deterministic correctness smoke passed |

## Required Shape Grid

| Family | Shapes |
| --- | --- |
| Decode | `M=1`, `K in {4096,12288}`, `N in {1024,4096,12288}` |

## Baseline Policy

- Correctness baseline: deterministic packed FP4 constant input and dequantized
  expected output.
- Readability baseline: PyTorch dequant plus matmul.
- Headline baseline: cuBLASLt/CUTLASS low-bit path or known strong FlashRT
  internal low-bit baseline where available.
- Keep all current claims labeled CUDA 12.8+ SM120 until another source path is
  added.

## Pending Results

Run after a built package artifact exists:

```bash
kernels benchmark flashrt/flashrt-smallm-gemm \
  --benchmark-script benchmarks/benchmark_nvfp4_w4a4_decode_matvec.py
```

Record:

| Workload | K | N | Mean ms | Ref ms | Speedup | Verified | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| pending | pending | pending | pending | pending | pending | pending | Built-artifact benchmark not run yet |

## Release Blockers

- Full `kernel-builder build` has not been run.
- HF benchmark runner has not been run against a built artifact.
- Fair low-bit vendor/library baseline is not recorded.
- Warpsplit small-M and tiny FP8 source slices are not exposed.
- Non-SM120 hardware validation is not applicable to the current v1 surface
  unless a non-SM120 source path is added.
