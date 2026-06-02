# Benchmark Results: flashrt-nvfp4

This file is the public result ledger for the v1 NVFP4 block. It is currently a
pre-release template plus local validation status, not a final release table.

Validated on June 2, 2026.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Built artifact: `torch211-cxx11-cu128-x86_64-linux`
- PyTorch inside HF testshell: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- Hardware scope: CUDA 12.8+ SM120 local validation only so far
- Benchmark path: local release-candidate runner over copied built artifact

## Current Scope

| API | Scope | Current status |
| --- | --- | --- |
| `nvfp4_sf_linear_to_swizzled` | CUTLASS Sm1xx NVFP4 scale-factor layout helper | Source accuracy full grid passed |

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

## Source Accuracy Gate

```bash
python scripts/accuracy_sweep.py --backend source --mode full --package flashrt-nvfp4
```

Result: passed 13 byte-parity checks over the required layout grid.

## Built Artifact Release-Candidate Results

Command:

```bash
python scripts/run_built_artifact_benchmarks.py \
  --package flashrt-nvfp4 --warmup 10 --iterations 50
```

| Workload | Mean ms | Ref ms | Speedup | Verified | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| `rows1_d4096` | 0.0074 | 0.5229 | 70.68x | yes | Python layout reference |
| `rows2_d4096` | 0.0073 | 0.9593 | 130.60x | yes | Python layout reference |
| `rows31_d4096` | 0.0074 | 14.2711 | 1939.87x | yes | Python layout reference |
| `rows32_d4096` | 0.0073 | 14.8109 | 2032.16x | yes | Python layout reference |
| `rows33_d4096` | 0.0074 | 15.2548 | 2054.55x | yes | Python layout reference |
| `rows127_d4096` | 0.0074 | 58.2877 | 7924.57x | yes | Python layout reference |
| `rows128_d4096` | 0.0075 | 58.1967 | 7763.73x | yes | Python layout reference |
| `rows129_d4096` | 0.0074 | 70.4571 | 9555.14x | yes | Python layout reference |
| `rows16_d1024` | 0.0073 | 1.9320 | 265.90x | yes | Python layout reference |
| `rows16_d2048` | 0.0073 | 3.7716 | 516.47x | yes | Python layout reference |
| `rows16_d8192` | 0.0074 | 21.1831 | 2847.88x | yes | Python layout reference |
| `rows16_d12288` | 0.0074 | 31.9058 | 4333.69x | yes | Python layout reference |
| `rows64_d16384` | 0.0074 | 133.1886 | 18031.32x | yes | Python layout reference |

## Release Blockers

- `torch211-cxx11-cu128-x86_64-linux` built artifact passed local
  release-candidate benchmark runner.
- Full `kernel-builder build-and-copy` matrix has not been run.
- Official Hub `kernels benchmark` has not been run after upload.
- Non-SM120 hardware validation is not applicable to the current v1 surface
  unless a non-SM120 source path is added.
