# Benchmark Results: flashrt-fused-quant

This file is the public result ledger for the v1 fused quantization block. It
is currently a pre-release template plus local validation status, not a final
release table.

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
| `silu_mul_quant_nvfp4_swizzled_bf16` | Split `gate, up` BF16 SwiGLU product plus NVFP4 swizzled quantization | Source synced, Tensor binding present, byte-parity smoke passed |
| `silu_mul_merged_quant_nvfp4_swizzled_bf16` | Merged `[gate | up]` BF16 SwiGLU product plus NVFP4 swizzled quantization | Source synced, Tensor binding present, byte-parity smoke passed |

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

## Pending Results

Run after a built package artifact exists:

```bash
kernels benchmark flashrt/flashrt-fused-quant \
  --benchmark-script benchmarks/benchmark_silu_mul_quant_nvfp4.py
```

Record:

| Workload | Shape | Mean ms | Ref ms | Speedup | GB/s | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| pending | pending | pending | pending | pending | pending | Built-artifact benchmark not run yet |

## Release Blockers

- Full `kernel-builder build` has not been run.
- HF benchmark runner has not been run against a built artifact.
- Memory-bandwidth benchmark results are not recorded.
- Residual/RMSNorm and SFA variants are not exposed.
- Multi-hardware validation is not complete.
