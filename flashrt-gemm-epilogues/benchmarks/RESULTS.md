# Benchmark Results: flashrt-gemm-epilogues

This file contains the current built-artifact release-candidate benchmark
results, followed by older source-extension triage notes.

Validated on June 2, 2026.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Driver: 580.82.07
- Built artifact: `torch211-cxx11-cu128-x86_64-linux`
- PyTorch inside HF testshell: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- Benchmark path: local release-candidate runner over copied built artifact

Timing method:

- `scripts/run_built_artifact_benchmarks.py`
- warmup 10, measured iterations 50
- reference timing uses the benchmark script's PyTorch eager reference

## Current Triage

- The FP8 quantization epilogue kernels are strong across the current shape
  suite on the built artifact: verified speedups range from 2.60x to 4.08x
  against the benchmark script PyTorch eager references.
- The BF16 GEMM epilogue wrapper is shape-sensitive. `M=1` is strong against
  the older source-extension `torch.addmm` baseline, but the public benchmark
  script is now performance-only for BF16 GEMM. Do not use BF16 GEMM as the v1
  headline claim.
- The GEMM path needs stronger baseline reporting. PyTorch eager is useful for
  HF benchmark readability, but serious GEMM claims should also compare against
  cuBLASLt or another vendor-library baseline.

## Source Accuracy Gate

```bash
python scripts/accuracy_sweep.py --backend source --mode full --package flashrt-gemm-epilogues
```

Result: passed 45 exact FP8 parity checks for:

- `bias_gelu_quantize_fp8_static_bf16`
- `gelu_quantize_fp8_static_bf16`
- `channel_scale_quantize_fp8_static_bf16`

BF16 GEMM epilogue wrappers are triage/compatibility APIs for v1 and are not
the headline correctness evidence in this package.

## Built Artifact Release-Candidate Results

Command:

```bash
python scripts/run_built_artifact_benchmarks.py \
  --package flashrt-gemm-epilogues --warmup 10 --iterations 50
```

Bias + GELU + FP8 quantization:

| Workload | Mean us | Ref us | Speedup | Verified |
| --- | ---: | ---: | ---: | --- |
| `decode_m1` | 7.25 | 28.93 | 3.99x | yes |
| `decode_m2` | 7.25 | 28.35 | 3.91x | yes |
| `decode_m4` | 7.54 | 30.80 | 4.08x | yes |
| `decode_m8` | 7.58 | 29.17 | 3.85x | yes |
| `small_m16` | 7.50 | 29.71 | 3.96x | yes |
| `small_m32` | 7.52 | 29.32 | 3.90x | yes |
| `prefill_m64` | 7.75 | 28.89 | 3.73x | yes |
| `prefill_m128` | 8.47 | 28.33 | 3.34x | yes |
| `prefill_m256` | 9.57 | 33.70 | 3.52x | yes |
| `wide_n8192_m16` | 7.34 | 28.95 | 3.95x | yes |
| `wide_n8192_m128` | 9.74 | 33.48 | 3.44x | yes |
| `vla_n12288_m16` | 7.70 | 29.08 | 3.78x | yes |
| `vla_n12288_m64` | 9.14 | 30.83 | 3.37x | yes |
| `vla_n16384_m16` | 7.72 | 29.47 | 3.82x | yes |
| `vla_n16384_m64` | 9.76 | 33.74 | 3.46x | yes |

GELU + FP8 quantization:

| Workload | Mean us | Ref us | Speedup | Verified |
| --- | ---: | ---: | ---: | --- |
| `decode_m1` | 7.24 | 21.16 | 2.92x | yes |
| `decode_m2` | 7.35 | 22.13 | 3.01x | yes |
| `decode_m4` | 7.48 | 21.90 | 2.93x | yes |
| `decode_m8` | 7.32 | 21.77 | 2.98x | yes |
| `small_m16` | 7.33 | 21.04 | 2.87x | yes |
| `small_m32` | 7.50 | 22.28 | 2.97x | yes |
| `prefill_m64` | 7.10 | 22.14 | 3.12x | yes |
| `prefill_m128` | 8.44 | 22.63 | 2.68x | yes |
| `prefill_m256` | 9.63 | 26.39 | 2.74x | yes |
| `wide_n8192_m16` | 7.39 | 21.45 | 2.90x | yes |
| `wide_n8192_m128` | 9.50 | 26.53 | 2.79x | yes |
| `vla_n12288_m16` | 7.43 | 22.58 | 3.04x | yes |
| `vla_n12288_m64` | 8.94 | 23.22 | 2.60x | yes |
| `vla_n16384_m16` | 7.60 | 21.96 | 2.89x | yes |
| `vla_n16384_m64` | 9.73 | 26.52 | 2.73x | yes |

Channel scale + FP8 quantization:

| Workload | Mean us | Ref us | Speedup | Verified |
| --- | ---: | ---: | ---: | --- |
| `decode_m1` | 7.31 | 26.05 | 3.56x | yes |
| `decode_m2` | 7.41 | 27.83 | 3.75x | yes |
| `decode_m4` | 7.44 | 26.69 | 3.59x | yes |
| `decode_m8` | 7.41 | 26.12 | 3.52x | yes |
| `small_m16` | 7.33 | 26.25 | 3.58x | yes |
| `small_m32` | 7.39 | 25.78 | 3.49x | yes |
| `prefill_m64` | 7.73 | 25.01 | 3.24x | yes |
| `prefill_m128` | 8.15 | 27.43 | 3.37x | yes |
| `prefill_m256` | 9.49 | 29.38 | 3.10x | yes |
| `wide_n8192_m16` | 7.37 | 26.28 | 3.56x | yes |
| `wide_n8192_m128` | 9.33 | 30.11 | 3.23x | yes |
| `vla_n12288_m16` | 7.40 | 25.27 | 3.41x | yes |
| `vla_n12288_m64` | 8.70 | 26.52 | 3.05x | yes |
| `vla_n16384_m16` | 7.64 | 25.66 | 3.36x | yes |
| `vla_n16384_m64` | 9.34 | 29.91 | 3.20x | yes |

BF16 GEMM epilogue latency-only compatibility measurements:

| Workload | Mean us | Notes |
| --- | ---: | --- |
| `bias_decode_m1` | 24.72 | performance-only |
| `gelu_decode_m1` | 25.38 | performance-only |
| `bias_decode_m8` | 25.28 | performance-only |
| `gelu_decode_m8` | 25.15 | performance-only |
| `bias_small_m16` | 25.97 | performance-only |
| `gelu_small_m16` | 26.19 | performance-only |
| `bias_prefill_m64` | 24.59 | performance-only |
| `gelu_prefill_m64` | 37.07 | performance-only |
| `bias_prefill_m128` | 37.66 | performance-only |
| `gelu_prefill_m128` | 36.29 | performance-only |
| `bias_wide_n8192_m16` | 39.76 | performance-only |
| `gelu_wide_n8192_m16` | 39.73 | performance-only |
| `bias_wide_k8192_m16` | 39.56 | performance-only |
| `gelu_wide_k8192_m16` | 39.37 | performance-only |

## GEMM Shape Suite

| API | Label | Shape | Fused us | Addmm ref us | Addmm speedup | TFLOPS | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bf16_gemm_bias` | `decode_m1` | `(1,4096,4096)` | 18.540 | 48.677 | 2.63x | 1.8 | promote |
| `bf16_gemm_bias_gelu` | `decode_m1` | `(1,4096,4096)` | 18.504 | 50.685 | 2.74x | 1.8 | promote |
| `bf16_gemm_bias` | `decode_m8` | `(8,4096,4096)` | 18.526 | 22.622 | 1.22x | 14.5 | reject |
| `bf16_gemm_bias_gelu` | `decode_m8` | `(8,4096,4096)` | 18.503 | 24.680 | 1.33x | 14.5 | reject |
| `bf16_gemm_bias` | `small_m16` | `(16,4096,4096)` | 22.393 | 22.632 | 1.01x | 24.0 | reject |
| `bf16_gemm_bias_gelu` | `small_m16` | `(16,4096,4096)` | 22.558 | 24.984 | 1.11x | 23.8 | reject |
| `bf16_gemm_bias` | `prefill_m64` | `(64,4096,4096)` | 18.909 | 36.913 | 1.95x | 113.6 | watch |
| `bf16_gemm_bias_gelu` | `prefill_m64` | `(64,4096,4096)` | 30.776 | 38.986 | 1.27x | 69.8 | reject |
| `bf16_gemm_bias` | `prefill_m128` | `(128,4096,4096)` | 32.816 | 38.982 | 1.19x | 130.9 | reject |
| `bf16_gemm_bias_gelu` | `prefill_m128` | `(128,4096,4096)` | 30.764 | 41.041 | 1.33x | 139.6 | reject |
| `bf16_gemm_bias` | `wide_n8192_m16` | `(16,8192,4096)` | 34.860 | 32.112 | 0.92x | 30.8 | reject |
| `bf16_gemm_bias_gelu` | `wide_n8192_m16` | `(16,8192,4096)` | 34.869 | 34.376 | 0.99x | 30.8 | reject |
| `bf16_gemm_bias` | `wide_k8192_m16` | `(16,4096,8192)` | 34.578 | 41.095 | 1.19x | 31.1 | reject |
| `bf16_gemm_bias_gelu` | `wide_k8192_m16` | `(16,4096,8192)` | 34.865 | 43.080 | 1.24x | 30.8 | reject |

## FP8 Quantization Shape Suite

This table uses the current default tile policy after sweeping
`FLASHRT_QUANT_BLOCK_SIZE=128|256|512|1024`.

| API | Label | Shape | Fused us | PyTorch eager us | Speedup | GB/s | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bias_gelu_quantize_fp8_static_bf16` | `decode_m1` | `(1,4096)` | 3.508 | 25.201 | 7.18x | 5.8 | promote |
| `gelu_quantize_fp8_static_bf16` | `decode_m1` | `(1,4096)` | 2.915 | 17.259 | 5.92x | 4.2 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `decode_m1` | `(1,4096)` | 2.985 | 21.028 | 7.05x | 6.9 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `decode_m2` | `(2,4096)` | 3.059 | 23.900 | 7.81x | 13.4 | promote |
| `gelu_quantize_fp8_static_bf16` | `decode_m2` | `(2,4096)` | 2.879 | 16.945 | 5.89x | 8.5 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `decode_m2` | `(2,4096)` | 2.929 | 20.707 | 7.07x | 14.0 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `decode_m4` | `(4,4096)` | 2.941 | 23.732 | 8.07x | 27.9 | promote |
| `gelu_quantize_fp8_static_bf16` | `decode_m4` | `(4,4096)` | 2.798 | 17.690 | 6.32x | 17.6 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `decode_m4` | `(4,4096)` | 3.047 | 20.708 | 6.80x | 26.9 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `decode_m8` | `(8,4096)` | 2.934 | 23.361 | 7.96x | 55.8 | promote |
| `gelu_quantize_fp8_static_bf16` | `decode_m8` | `(8,4096)` | 2.757 | 16.398 | 5.95x | 35.7 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `decode_m8` | `(8,4096)` | 2.766 | 19.370 | 7.00x | 59.2 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `small_m16` | `(16,4096)` | 2.820 | 22.674 | 8.04x | 116.2 | promote |
| `gelu_quantize_fp8_static_bf16` | `small_m16` | `(16,4096)` | 2.704 | 16.794 | 6.21x | 72.7 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `small_m16` | `(16,4096)` | 2.790 | 19.662 | 7.05x | 117.5 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `small_m32` | `(32,4096)` | 2.840 | 22.387 | 7.88x | 230.8 | promote |
| `gelu_quantize_fp8_static_bf16` | `small_m32` | `(32,4096)` | 2.757 | 16.529 | 5.99x | 142.6 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `small_m32` | `(32,4096)` | 2.737 | 19.421 | 7.10x | 239.5 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `prefill_m64` | `(64,4096)` | 2.789 | 23.895 | 8.57x | 469.9 | promote |
| `gelu_quantize_fp8_static_bf16` | `prefill_m64` | `(64,4096)` | 2.850 | 16.847 | 5.91x | 275.9 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `prefill_m64` | `(64,4096)` | 2.766 | 19.724 | 7.13x | 473.8 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `prefill_m128` | `(128,4096)` | 4.304 | 23.708 | 5.51x | 609.1 | promote |
| `gelu_quantize_fp8_static_bf16` | `prefill_m128` | `(128,4096)` | 4.294 | 16.740 | 3.90x | 366.3 | watch |
| `channel_scale_quantize_fp8_static_bf16` | `prefill_m128` | `(128,4096)` | 4.308 | 20.082 | 4.66x | 608.5 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `prefill_m256` | `(256,4096)` | 4.431 | 29.135 | 6.58x | 1183.3 | promote |
| `gelu_quantize_fp8_static_bf16` | `prefill_m256` | `(256,4096)` | 4.355 | 22.117 | 5.08x | 722.3 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `prefill_m256` | `(256,4096)` | 4.354 | 25.501 | 5.86x | 1204.2 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `wide_n8192_m16` | `(16,8192)` | 2.969 | 22.863 | 7.70x | 220.7 | promote |
| `gelu_quantize_fp8_static_bf16` | `wide_n8192_m16` | `(16,8192)` | 2.771 | 17.482 | 6.31x | 141.9 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `wide_n8192_m16` | `(16,8192)` | 2.794 | 19.597 | 7.01x | 234.5 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `wide_n8192_m128` | `(128,8192)` | 4.437 | 29.439 | 6.63x | 1181.5 | promote |
| `gelu_quantize_fp8_static_bf16` | `wide_n8192_m128` | `(128,8192)` | 4.447 | 21.818 | 4.91x | 707.5 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `wide_n8192_m128` | `(128,8192)` | 4.443 | 26.027 | 5.86x | 1180.1 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `vla_n12288_m16` | `(16,12288)` | 2.845 | 22.829 | 8.03x | 345.6 | promote |
| `gelu_quantize_fp8_static_bf16` | `vla_n12288_m16` | `(16,12288)` | 2.954 | 16.667 | 5.64x | 199.7 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `vla_n12288_m16` | `(16,12288)` | 2.817 | 19.493 | 6.92x | 348.9 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `vla_n12288_m64` | `(64,12288)` | 4.390 | 26.320 | 6.00x | 895.7 | promote |
| `gelu_quantize_fp8_static_bf16` | `vla_n12288_m64` | `(64,12288)` | 4.292 | 17.924 | 4.18x | 549.7 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `vla_n12288_m64` | `(64,12288)` | 4.342 | 22.480 | 5.18x | 905.7 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `vla_n16384_m16` | `(16,16384)` | 2.828 | 22.864 | 8.08x | 463.4 | promote |
| `gelu_quantize_fp8_static_bf16` | `vla_n16384_m16` | `(16,16384)` | 2.699 | 16.744 | 6.20x | 291.4 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `vla_n16384_m16` | `(16,16384)` | 2.797 | 19.592 | 7.01x | 468.7 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `vla_n16384_m64` | `(64,16384)` | 4.436 | 29.694 | 6.69x | 1182.0 | promote |
| `gelu_quantize_fp8_static_bf16` | `vla_n16384_m64` | `(64,16384)` | 4.438 | 21.919 | 4.94x | 708.9 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `vla_n16384_m64` | `(64,16384)` | 4.364 | 25.660 | 5.88x | 1201.4 | promote |

First use of a new `(M,N,K,epilogue)` GEMM shape pays an autotune cost. Later
calls reuse the cached algorithm.

## Next Benchmark Work

- Add cuBLASLt/vendor-library baseline reporting for GEMM epilogue shapes.
- Investigate tile/algo policy for weak GEMM shapes before making broad public
  claims.
- Re-run the official Hub `kernels benchmark` after upload.
- Validate the FP8 quantization table on non-SM120 hardware before making a
  broad CUDA hardware claim.
