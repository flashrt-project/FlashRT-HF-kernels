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
  suite after row/column tile policy tuning.
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

| API | Label | Shape | Fused us | PyTorch eager us | Speedup | GB/s | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bias_gelu_quantize_fp8_static_bf16` | `decode_m1` | `(1,4096)` | 2.704 | 22.571 | 8.35x | 7.6 | promote |
| `gelu_quantize_fp8_static_bf16` | `decode_m1` | `(1,4096)` | 2.590 | 16.482 | 6.36x | 4.7 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `decode_m1` | `(1,4096)` | 2.692 | 19.404 | 7.21x | 7.6 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `decode_m8` | `(8,4096)` | 2.601 | 21.847 | 8.40x | 63.0 | promote |
| `gelu_quantize_fp8_static_bf16` | `decode_m8` | `(8,4096)` | 2.558 | 16.116 | 6.30x | 38.4 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `decode_m8` | `(8,4096)` | 2.619 | 19.176 | 7.32x | 62.5 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `small_m16` | `(16,4096)` | 2.610 | 21.995 | 8.43x | 125.5 | promote |
| `gelu_quantize_fp8_static_bf16` | `small_m16` | `(16,4096)` | 2.573 | 16.253 | 6.32x | 76.4 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `small_m16` | `(16,4096)` | 2.588 | 19.234 | 7.43x | 126.6 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `prefill_m64` | `(64,4096)` | 2.620 | 22.226 | 8.48x | 500.3 | promote |
| `gelu_quantize_fp8_static_bf16` | `prefill_m64` | `(64,4096)` | 2.607 | 16.785 | 6.44x | 301.7 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `prefill_m64` | `(64,4096)` | 2.557 | 19.490 | 7.62x | 512.5 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `prefill_m128` | `(128,4096)` | 4.111 | 22.435 | 5.46x | 637.7 | promote |
| `gelu_quantize_fp8_static_bf16` | `prefill_m128` | `(128,4096)` | 4.116 | 16.640 | 4.04x | 382.1 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `prefill_m128` | `(128,4096)` | 4.116 | 20.460 | 4.97x | 636.9 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `wide_n8192_m16` | `(16,8192)` | 2.613 | 22.501 | 8.61x | 250.8 | promote |
| `gelu_quantize_fp8_static_bf16` | `wide_n8192_m16` | `(16,8192)` | 2.639 | 16.288 | 6.17x | 149.0 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `wide_n8192_m16` | `(16,8192)` | 2.738 | 19.417 | 7.09x | 239.4 | promote |
| `bias_gelu_quantize_fp8_static_bf16` | `wide_n8192_m128` | `(128,8192)` | 4.132 | 29.695 | 7.19x | 1268.7 | promote |
| `gelu_quantize_fp8_static_bf16` | `wide_n8192_m128` | `(128,8192)` | 4.132 | 21.438 | 5.19x | 761.3 | promote |
| `channel_scale_quantize_fp8_static_bf16` | `wide_n8192_m128` | `(128,8192)` | 4.121 | 25.748 | 6.25x | 1272.4 | promote |

First use of a new `(M,N,K,epilogue)` GEMM shape pays an autotune cost. Later
calls reuse the cached algorithm.

## Next Benchmark Work

- Add cuBLASLt/vendor-library baseline reporting for GEMM epilogue shapes.
- Investigate tile/algo policy for weak GEMM shapes before making broad public
  claims.
- Continue tile policy tuning to raise the lower-bound throughput for large
  `M=128,N=4096` FP8 quantization shapes.
- Add VLA/video-specific projection shape groups once extracted from real
  FlashRT traces.
- Compare against the uploaded HF kernel artifact once the package is uploaded.
