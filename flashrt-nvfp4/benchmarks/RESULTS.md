# Benchmark Results: flashrt-nvfp4

This file is the public result ledger for the v1 NVFP4 block. It is currently a
pre-release template plus local validation status, not a final release table.

Validated on June 2, 2026.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- Hardware scope: SM120 local validation only so far
- Benchmark path: pending built package artifact

## Current Scope

| API | Scope | Current status |
| --- | --- | --- |
| `nvfp4_sf_linear_to_swizzled` | CUTLASS Sm1xx NVFP4 scale-factor layout helper | Source synced, Tensor binding present, tests and benchmark harness present |

## Required Shape Grid

| Family | Shapes |
| --- | --- |
| Layout boundary | rows `1,2,31,32,33,127,128,129`, D `4096` |
| Contracted dimension | rows `16`, D `1024,2048,8192,12288`; rows `64`, D `16384` |

## Baseline Policy

- Correctness baseline: byte-for-byte swizzle reference.
- Performance baseline: PyTorch/CUDA tensor reshape path for readability.
- Headline fused GEMM claims are not allowed from this package until a fused
  NVFP4 GEMM epilogue surface is added and compared against CUTLASS/cuBLASLt or
  an unfused strong CUDA chain.

## Pending Results

Run after a built package artifact exists:

```bash
kernels benchmark flashrt/flashrt-nvfp4 \
  --benchmark-script benchmarks/benchmark_nvfp4_sf_reshape.py
```

Record:

| Workload | Mean ms | Ref ms | Speedup | Verified | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| pending | pending | pending | pending | pending | Built-artifact benchmark not run yet |

## Release Blockers

- Full `kernel-builder build` has not been run.
- HF benchmark runner has not been run against a built artifact.
- Multi-hardware validation is not complete.
