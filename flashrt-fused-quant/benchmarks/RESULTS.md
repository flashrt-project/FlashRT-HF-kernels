# Benchmark Results: flashrt-fused-quant

This file is the public result ledger for the v1 fused quantization block. It
is currently a pre-release template plus local validation status, not a final
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
| `silu_mul_quant_nvfp4_swizzled_bf16` | Split `gate, up` BF16 SwiGLU product plus NVFP4 swizzled quantization | Source accuracy full grid passed |
| `silu_mul_merged_quant_nvfp4_swizzled_bf16` | Merged `[gate | up]` BF16 SwiGLU product plus NVFP4 swizzled quantization | Source accuracy full grid passed |

## Required Shape Grid

| Family | Shapes |
| --- | --- |
| Decode FFN | rows `1,2,4,8`, hidden `4096,8192,12288,16384` |
| Small batch FFN | rows `16,32`, hidden `4096,8192,12288,16384` |
| Prefill/video | rows `64,128,256,1024,2520`, hidden `4096,8192,12288` |

## Baseline Policy

- Correctness baseline: package-local fake-quant reference with packed FP4 and
  swizzled scale-factor byte parity.
- Readability baseline: PyTorch eager `SiLU(gate) * up` plus fake quantization.
- Headline metric: latency and effective memory bandwidth.
- Multi-output correctness stays in package tests because the HF benchmark
  runner verifies one tensor output.

## Source Accuracy Gate

```bash
python scripts/accuracy_sweep.py --backend source --mode full --package flashrt-fused-quant
```

Result: passed 144 checks. Both split and merged APIs passed packed output and
swizzled scale byte parity over the required shape grid.

## Built Artifact Release-Candidate Results

Command:

```bash
python scripts/run_built_artifact_benchmarks.py \
  --package flashrt-fused-quant --warmup 10 --iterations 50
```

The current public benchmark script is latency-only because the APIs return
both packed data and swizzled scales, while the HF runner verifies one output
tensor. Multi-output byte parity is covered by `accuracy_sweep.py`.
`torch.compile` status is `no_reference` for this public benchmark until a
tensor-only reference path is added. Do not publish eager or compile speedups
from this table.

| API | Decode 4096 us | Decode 8192 us | Decode 12288 us | Decode 16384 us | Video 2520 x 12288 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| split | 12.90-13.18 | 18.48-18.84 | 24.49-26.10 | 29.94-30.10 | 171.02 |
| merged | 12.82-12.92 | 18.53-18.57 | 23.86-24.20 | 29.33-29.91 | 168.46 |

Selected full-shape latencies:

| API | Workload | Mean us |
| --- | --- | ---: |
| split | `small_r16_h4096` | 12.77 |
| split | `small_r32_h8192` | 18.82 |
| split | `prefill_r256_h12288` | 26.10 |
| split | `video_r1024_h4096` | 22.36 |
| split | `video_r2520_h12288` | 171.02 |
| merged | `small_r16_h4096` | 12.95 |
| merged | `small_r32_h8192` | 18.68 |
| merged | `prefill_r256_h12288` | 25.64 |
| merged | `video_r1024_h4096` | 22.25 |
| merged | `video_r2520_h12288` | 168.46 |

## Release Blockers

- Local release-candidate benchmark runner has been run against the built
  artifact. Official Hub `kernels benchmark` has not been run after upload.
- Public benchmark needs a `verify_*`/reference path before reporting eager or
  `torch.compile` speedups.
- Memory-bandwidth benchmark results are not recorded yet.
- Residual/RMSNorm and SFA variants are not exposed.
- Non-SM120 hardware validation is not applicable to the current v1 surface
  unless a non-SM120 source path is added.
