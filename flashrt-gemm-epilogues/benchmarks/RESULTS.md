# Benchmark Results: flashrt-gemm-epilogues

These are preliminary local numbers. They are useful for prioritizing kernel
work, but they are not yet a stable release benchmark table.

Validated on June 2, 2026.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- Benchmark path: local source extension loaded through
  `torch.utils.cpp_extension`

Timing method:

- CUDA synchronized wall-clock timing
- 10-20 warmup iterations depending on the sweep
- 50-100 iterations for GEMM
- 100-200 iterations for quantization kernels
- Status threshold used for triage:
  - GEMM epilogue: `promote` at `>=2.0x` against `torch.addmm` or
    `gelu(torch.addmm)`, `watch` at `>=1.5x`
  - FP8 quantization: `promote` at `>=4.0x`, `watch` at `>=1.5x`

## Current Triage

- The FP8 quantization epilogue kernels are strong across the current shape
  suite after row/column tile policy tuning. The only current watch item in
  the default-policy table is `gelu_quantize_fp8_static_bf16` at
  `(M,N)=(128,4096)`, which measures 3.90x against PyTorch eager.
- The BF16 GEMM epilogue wrapper is shape-sensitive. `M=1` is strong against
  the stricter `torch.addmm` baseline. `M=64` bias is close but remains a
  watch shape. `M=8`, `M=16`, `M=128`, and the current wider projection shapes
  should not be promoted as headline shapes yet.
- The GEMM path needs stronger baseline reporting. PyTorch eager is useful for
  HF benchmark readability, but serious GEMM claims should also compare against
  cuBLASLt or another vendor-library baseline.

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
- Re-run the HF benchmark CLI against a built or uploaded kernel artifact.
- Validate the FP8 quantization table on non-SM120 hardware before making a
  broad CUDA hardware claim.
