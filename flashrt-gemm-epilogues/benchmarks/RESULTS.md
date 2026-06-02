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
- 20 warmup iterations
- 100 iterations for GEMM
- 200 iterations for quantization kernels

## Results

| API | Shape | Fused us | PyTorch eager us | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `bf16_gemm_bias_gelu` | `(M,N,K)=(64,4096,4096)` | 30.813 | 42.530 | 1.38x |
| `bf16_gemm_bias` | `(M,N,K)=(64,4096,4096)` | 18.506 | 40.565 | 2.19x |
| `bias_gelu_quantize_fp8_static_bf16` | `(64,4096)` | 4.084 | 22.729 | 5.57x |
| `channel_scale_quantize_fp8_static_bf16` | `(64,4096)` | 2.464 | 19.581 | 7.95x |

## Interpretation

- The FP8 quantization epilogue kernels are already strong on the local 5090
  environment.
- The BF16 GEMM epilogue path now uses per-shape cuBLASLt algorithm autotuning
  and caching. The autotuned M=64 path removes the previous `GELU_BIAS`
  regression.
- First use of a new `(M,N,K,epilogue)` shape pays a small autotune cost. Later
  calls reuse the cached algorithm.

## GEMM Shape Sweep

Same environment and local source-extension path. This sweep uses
`N=K=4096` and varies `M`.

| API | Shape | Fused us | PyTorch eager us | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `bf16_gemm_bias` | `(1,4096,4096)` | 18.480 | 50.685 | 2.74x |
| `bf16_gemm_bias_gelu` | `(1,4096,4096)` | 18.463 | 52.610 | 2.85x |
| `bf16_gemm_bias` | `(8,4096,4096)` | 18.470 | 24.633 | 1.33x |
| `bf16_gemm_bias_gelu` | `(8,4096,4096)` | 18.467 | 26.679 | 1.44x |
| `bf16_gemm_bias` | `(16,4096,4096)` | 22.374 | 24.623 | 1.10x |
| `bf16_gemm_bias_gelu` | `(16,4096,4096)` | 22.543 | 26.823 | 1.19x |
| `bf16_gemm_bias` | `(64,4096,4096)` | 18.506 | 40.565 | 2.19x |
| `bf16_gemm_bias_gelu` | `(64,4096,4096)` | 30.813 | 42.530 | 1.38x |
| `bf16_gemm_bias` | `(128,4096,4096)` | 32.880 | 34.968 | 1.06x |
| `bf16_gemm_bias_gelu` | `(128,4096,4096)` | 30.822 | 36.998 | 1.20x |

The previous `M=64` GELU regression was caused by using only the first
cuBLASLt heuristic result. Benchmarking multiple candidate algorithms and
caching the fastest candidate fixes the outlier on the local test GPU.

## Next Benchmark Work

- Run a matrix of decode and prefill shapes: small M, medium M, and large M.
- Separate GEMM time from epilogue time where possible.
- Compare against the uploaded HF kernel artifact once the package is uploaded
  to a Hub namespace.
- Add CI-friendly smoke benchmarks with very small iteration counts.
