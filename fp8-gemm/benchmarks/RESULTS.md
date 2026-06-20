# Benchmark Results: fp8-gemm

Validated locally on June 20, 2026.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- Build target: `sm_120a`
- Backend: source extension
- Benchmark command:

```bash
python fp8-gemm/benchmarks/benchmark.py \
  --backend source --mode headline --warmup 20 --iterations 100 --compile-ref
```

Correctness gate:

```bash
python fp8-gemm/tests/test_fp8_gemm.py --backend source --mode full
```

Result: 8/8 rows passed. Metrics recorded: max absolute error, mean absolute
error, p99 absolute error, cosine similarity, dtype, and tolerance. Public v1
scope is `M=1` decode and `2 <= M <= 64` small-M rows. M=128 is not exposed in
v1 because it needs separate performance-positive tile tuning.

## Headline Rows

| Shape | Tile | FlashRT us | Torch eager us | Torch compile us | Speedup vs eager | Speedup vs compile | Max abs | P99 abs | Cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `M=1,K=4096,N=2048` | `gemv_fp8_m1_w4` | 6.188 | 32.799 | 41.714 | 5.30x | 6.74x | 0.000 | 0.000 | 1.000000 |
| `M=1,K=4096,N=8192` | `gemv_fp8_m1_w8` | 10.290 | 162.342 | 156.012 | 15.78x | 15.16x | 0.000 | 0.000 | 1.000000 |
| `M=16,K=4096,N=4096` | `ld_fp8_gemm_16x128x256_w4` | 14.391 | 106.215 | 96.204 | 7.38x | 6.68x | 0.000 | 0.000 | 1.000000 |
| `M=32,K=4096,N=8192` | `ld_fp8_gemm_32x128x256_w4` | 22.581 | 200.997 | 189.331 | 8.90x | 8.38x | 0.000 | 0.000 | 1.000000 |
| `M=64,K=512,N=1024` | `ld_fp8_gemm_64x128x256_w4` | 8.259 | 18.085 | 50.002 | 2.19x | 6.05x | 0.000 | 0.000 | 1.000000 |

## M=1 Variant Sweep

The dispatcher defaults to `variant=0`. Explicit variants are retained for
benchmarking and tuning; public callers should use `variant=0` unless they have
measured their exact shape.

| Shape | Variant | Tile | FlashRT us | Speedup vs eager | Status |
| --- | ---: | --- | ---: | ---: | --- |
| `M=1,K=4096,N=2048` | 0 | `gemv_fp8_m1_w4` | 6.188 | 5.30x | pass |
| `M=1,K=4096,N=2048` | 4 | `gemv_fp8_m1_w4` | 6.186 | 5.30x | pass |
| `M=1,K=4096,N=2048` | 8 | `gemv_fp8_m1_w8` | 6.184 | 5.30x | pass |
| `M=1,K=4096,N=2048` | 16 | `gemv_fp8_m1_w16` | 6.188 | 5.30x | pass |
| `M=1,K=4096,N=8192` | 0 | `gemv_fp8_m1_w8` | 10.290 | 15.78x | pass |
| `M=1,K=4096,N=8192` | 4 | `gemv_fp8_m1_w4` | 10.274 | 15.81x | pass |
| `M=1,K=4096,N=8192` | 8 | `gemv_fp8_m1_w8` | 10.272 | 15.82x | pass |
| `M=1,K=4096,N=8192` | 16 | `gemv_fp8_m1_w16` | 10.278 | 15.80x | pass |

## Release Status

- Source correctness: passed.
- Source benchmark/tile sweep: passed for v1 public scope.
- Installed-artifact correctness: pending after HF Jobs build.
- Hub artifact benchmark: pending after upload.
