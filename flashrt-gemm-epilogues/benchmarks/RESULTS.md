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
  - GEMM epilogue: `promote` at `>=2.0x`, `watch` at `>=1.5x`
  - FP8 quantization: `promote` at `>=4.0x`, `watch` at `>=1.5x`

## Current Triage

- The FP8 quantization epilogue kernels are consistently strong across the
  current shape suite.
- The BF16 GEMM epilogue wrapper is shape-sensitive. `M=1` and `M=64` bias
  are strong against PyTorch eager, but `M=8`, `M=16`, and `M=128` should not
  be promoted as headline shapes yet.
- The GEMM path needs stronger baseline reporting. PyTorch eager is useful for
  HF benchmark readability, but serious GEMM claims should also compare against
  cuBLASLt or another vendor-library baseline.

## GEMM Shape Suite

| API | Label | Shape | Fused us | PyTorch eager us | Speedup | TFLOPS | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bf16_gemm_bias` | `decode_m1` | `(1,4096,4096)` | 18.627 | 51.244 | 2.75x | 1.8 | promote |
| `bf16_gemm_bias_gelu` | `decode_m1` | `(1,4096,4096)` | 18.578 | 53.046 | 2.86x | 1.8 | promote |
| `bf16_gemm_bias` | `decode_m8` | `(8,4096,4096)` | 18.621 | 24.769 | 1.33x | 14.4 | reject |
| `bf16_gemm_bias_gelu` | `decode_m8` | `(8,4096,4096)` | 18.579 | 26.803 | 1.44x | 14.4 | reject |
| `bf16_gemm_bias` | `small_m16` | `(16,4096,4096)` | 22.376 | 24.775 | 1.11x | 24.0 | reject |
| `bf16_gemm_bias_gelu` | `small_m16` | `(16,4096,4096)` | 22.633 | 27.276 | 1.21x | 23.7 | reject |
| `bf16_gemm_bias` | `prefill_m64` | `(64,4096,4096)` | 18.570 | 40.937 | 2.20x | 115.6 | promote |
| `bf16_gemm_bias_gelu` | `prefill_m64` | `(64,4096,4096)` | 30.918 | 42.427 | 1.37x | 69.5 | reject |
| `bf16_gemm_bias` | `prefill_m128` | `(128,4096,4096)` | 32.880 | 34.962 | 1.06x | 130.6 | reject |
| `bf16_gemm_bias_gelu` | `prefill_m128` | `(128,4096,4096)` | 30.833 | 37.004 | 1.20x | 139.3 | reject |
| `bf16_gemm_bias` | `wide_n8192_m16` | `(16,8192,4096)` | 34.925 | 58.506 | 1.68x | 30.7 | watch |
| `bf16_gemm_bias_gelu` | `wide_n8192_m16` | `(16,8192,4096)` | 34.911 | 60.822 | 1.74x | 30.8 | watch |
| `bf16_gemm_bias` | `wide_k8192_m16` | `(16,4096,8192)` | 34.566 | 43.203 | 1.25x | 31.1 | reject |
| `bf16_gemm_bias_gelu` | `wide_k8192_m16` | `(16,4096,8192)` | 34.917 | 45.205 | 1.29x | 30.8 | reject |

## FP8 Quantization Shape Suite

| API | Label | Shape | Fused us | PyTorch eager us | Speedup | GB/s | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bias_gelu_quantize_fp8_static_bf16` | `decode_m1` | `(1,4096)` | 2.520 | 30.416 | 12.07x | 8.1 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `decode_m1` | `(1,4096)` | 3.050 | 25.989 | 8.52x | 6.7 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `decode_m8` | `(8,4096)` | 4.126 | 29.525 | 7.16x | 39.7 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `decode_m8` | `(8,4096)` | 3.238 | 24.806 | 7.66x | 50.6 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `small_m16` | `(16,4096)` | 2.891 | 26.492 | 9.16x | 113.3 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `small_m16` | `(16,4096)` | 2.825 | 25.250 | 8.94x | 116.0 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `prefill_m64` | `(64,4096)` | 4.105 | 29.385 | 7.16x | 319.3 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `prefill_m64` | `(64,4096)` | 3.305 | 21.547 | 6.52x | 396.5 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `prefill_m128` | `(128,4096)` | 4.135 | 25.951 | 6.28x | 634.0 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `prefill_m128` | `(128,4096)` | 4.122 | 20.484 | 4.97x | 635.9 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `wide_n8192_m16` | `(16,8192)` | 2.582 | 24.660 | 9.55x | 253.8 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `wide_n8192_m16` | `(16,8192)` | 2.576 | 19.043 | 7.39x | 254.4 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `wide_n8192_m128` | `(128,8192)` | 6.183 | 29.472 | 4.77x | 848.0 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `wide_n8192_m128` | `(128,8192)` | 4.149 | 25.826 | 6.22x | 1263.5 | promote |

First use of a new `(M,N,K,epilogue)` GEMM shape pays an autotune cost. Later
calls reuse the cached algorithm.

## Next Benchmark Work

- Add cuBLASLt/vendor-library baseline reporting for GEMM epilogue shapes.
- Investigate tile/algo policy for weak GEMM shapes before making broad public
  claims.
- Add VLA/video-specific projection shape groups once extracted from real
  FlashRT traces.
- Compare against the uploaded HF kernel artifact once the package is uploaded.
