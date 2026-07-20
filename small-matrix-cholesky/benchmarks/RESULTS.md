# Benchmark results

## NVIDIA A800 source extension

- Date: 2026-07-19
- GPU: NVIDIA A800-SXM4-80GB, compute capability 8.0
- PyTorch: 2.7.1+cu126
- CUDA reported by PyTorch: 12.6
- Warmup: 10 iterations
- Samples: 50 iterations, median CUDA-event latency
- Candidate and PyTorch baseline both use preallocated output tensors

| Batch | N | Candidate (ms) | PyTorch (ms) | Speedup | Candidate TFLOP/s | I/O GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 32 | 0.107712 | 0.231328 | 2.148x | 0.415 | 311.520 |
| 1024 | 64 | 0.229408 | 0.269536 | 1.175x | 0.390 | 146.265 |
| 256 | 128 | 0.234016 | 0.326560 | 1.395x | 0.765 | 143.385 |

- Candidate geometric mean: 0.179490 ms
- PyTorch geometric mean: 0.273067 ms
- Geometric-mean speedup: 1.521x

These are package-specific results without input/result memoization. They are
not the GPU MODE B200 leaderboard score.

## NVIDIA RTX 5090 maintainer promotion run

- Date: 2026-07-20
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0
- PyTorch: 2.11.0+cu128
- Warmup: 20 iterations
- Samples: 100 iterations, median CUDA-event latency
- All paths use preallocated output tensors. The compile baseline is the same
  `torch.linalg.cholesky_ex(..., out=...)` reference under
  `torch.compile(fullgraph=True)`.

| Batch | N | Candidate (ms) | PyTorch eager (ms) | PyTorch compile (ms) | vs eager | vs compile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 32 | 0.042272 | 0.107200 | 0.121136 | 2.536x | 2.866x |
| 1024 | 64 | 0.088736 | 0.145872 | 0.151888 | 1.644x | 1.712x |
| 256 | 128 | 0.093888 | 0.212496 | 0.202016 | 2.263x | 2.152x |

- Candidate geometric mean: 0.070619 ms
- PyTorch eager geometric mean: 0.149224 ms
- PyTorch compile geometric mean: 0.154903 ms
- Geometric-mean speedup: 2.113x vs eager, 2.194x vs compile
