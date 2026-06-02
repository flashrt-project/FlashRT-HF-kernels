# Benchmark Results: flashrt-smallm-gemm

This file is the public result ledger for the v1 small-M GEMM block. It is
currently a pre-release template plus local validation status, not a final
release table.

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
| `nvfp4_w4a4_decode_matvec_bf16out` | SM120 NVFP4 W4A4 M=1 decode matvec with BF16 output | Source accuracy full grid passed |

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

## Source Accuracy Gate

```bash
python scripts/accuracy_sweep.py --backend source --mode full --package flashrt-smallm-gemm
```

Result: passed 12 checks. The sweep covers constant inputs and random
packed/dequantized references over `K in {4096,12288}` and
`N in {1024,4096,12288}`. Source sweep measured BF16 output `max_ulp <= 4`;
the built artifact release gate is `max_ulp <= 5`.

## Built Artifact Release-Candidate Results

Command:

```bash
python scripts/run_built_artifact_benchmarks.py \
  --package flashrt-smallm-gemm --warmup 10 --iterations 50
```

The benchmark uses random packed W4A4 inputs and a PyTorch readability
baseline that dequantizes FP4 values/scales and performs the matvec. The older
constant-input table compared against `torch.full` and is invalidated.

| Workload | K | N | Mean ms | Ref ms | Speedup | Verified | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `k4096_n1024` | 4096 | 1024 | 0.0331 | 0.4672 | 14.12x | yes | random/dequant baseline |
| `k4096_n4096` | 4096 | 4096 | 0.1078 | 1.4832 | 13.76x | yes | random/dequant baseline |
| `k4096_n12288` | 4096 | 12288 | 0.2596 | 4.1848 | 16.12x | yes | random/dequant baseline |
| `k12288_n1024` | 12288 | 1024 | 0.0844 | 0.5472 | 6.48x | yes | random/dequant baseline |
| `k12288_n4096` | 12288 | 4096 | 0.3080 | 1.8043 | 5.86x | yes | random/dequant baseline |
| `k12288_n12288` | 12288 | 12288 | 0.7575 | 5.0800 | 6.71x | yes | random/dequant baseline |

## Release Blockers

- `torch211-cxx11-cu128-x86_64-linux` built artifact passed installed
  accuracy sweep.
- Full `kernel-builder build-and-copy` matrix has not been run.
- Local release-candidate benchmark runner has been run against the built
  artifact with random/dequant baselines. Official Hub `kernels benchmark` has
  not been run after upload.
- Fair low-bit vendor/library baseline is not recorded.
- Warpsplit small-M and tiny FP8 source slices are not exposed.
- Non-SM120 hardware validation is not applicable to the current v1 surface
  unless a non-SM120 source path is added.
