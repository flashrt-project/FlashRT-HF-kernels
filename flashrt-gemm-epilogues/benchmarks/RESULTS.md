# Benchmark Results: flashrt-gemm-epilogues

These are preliminary local numbers. They are useful for prioritizing kernel
work, but they are not yet a stable release benchmark table.

Validated on June 1, 2026.

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
| `bf16_gemm_bias_gelu` | `(M,N,K)=(64,4096,4096)` | 60.008 | 42.174 | 0.70x |
| `bf16_gemm_bias` | `(M,N,K)=(64,4096,4096)` | 36.619 | 40.652 | 1.11x |
| `bias_gelu_quantize_fp8_static_bf16` | `(64,4096)` | 4.084 | 22.729 | 5.57x |
| `channel_scale_quantize_fp8_static_bf16` | `(64,4096)` | 2.464 | 19.581 | 7.95x |

## Interpretation

- The FP8 quantization epilogue kernels are already strong on the local 5090
  environment.
- `bf16_gemm_bias` is slightly faster than the eager BF16 GEMM plus bias path
  for this shape.
- `bf16_gemm_bias_gelu` is slower than PyTorch eager for this shape. Treat this
  as a tuning target before promoting it as a performance win. Likely next
  checks are cuBLASLt heuristic selection, workspace size, and a broader shape
  sweep.

## Next Benchmark Work

- Run a matrix of decode and prefill shapes: small M, medium M, and large M.
- Separate GEMM time from epilogue time where possible.
- Compare against the uploaded HF kernel artifact once the package is uploaded
  to a Hub namespace.
- Add CI-friendly smoke benchmarks with very small iteration counts.
