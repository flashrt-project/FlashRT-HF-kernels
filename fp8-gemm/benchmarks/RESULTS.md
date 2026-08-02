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
SM120 scope is `M=1` decode and `2 <= M <= 64` small-M rows. SM110 uses a
separate full-row CUTLASS dispatcher described below.

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

## Block-128 Scaled GEMM

Measured on RTX 5090 against an independent binding of the original FlashRT
pointer API. The wrapper and native columns execute the same production
CUTLASS kernel from separate extension modules. PyTorch eager and compile
dequantize the block-scaled tensors to FP32, run the GEMM, and cast to BF16.
CUTLASS is already the native implementation, so there is no additional
contract-equivalent library row.

| Workload `(M,K,N)` | Native us | Wrapper us | Wrapper/native | Eager us | Compile us | Max abs | P99 abs | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode `(1,1024,1024)` | 10.287 | 10.276 | 0.999 | 36.256 | 42.242 | 0.000000 | 0.000000 | 1.0000001 |
| action `(51,1536,1536)` | 14.364 | 14.360 | 1.000 | 55.595 | 51.500 | 0.000061 | 0.000000 | 1.0000001 |
| GROOT `(277,2048,2048)` | 28.698 | 28.692 | 1.000 | 104.895 | 73.989 | 0.000122 | 0.000000 | 1.0000001 |
| vision `(1024,1152,1152)` | 18.456 | 18.466 | 1.001 | 101.103 | 79.565 | 0.000122 | 0.000000 | 1.0000000 |
| video `(2520,3072,3072)` | 114.188 | 114.696 | 1.004 | 1032.687 | 930.219 | 0.000244 | 0.000000 | 1.0000000 |
| Qwen MLP `(128,4096,12288)` | 51.258 | 51.253 | 1.000 | 892.744 | 390.698 | 0.000244 | 0.000000 | 1.0000000 |

All wrapper outputs were bitwise equal to the original native entry. The
PyTorch-reference metrics above use production-scale ranges; the wider full
correctness sweep remains the release gate.

## Release Status

- Source correctness: passed.
- Source benchmark/tile sweep: passed for v1 public scope.
- Existing SM120 installed artifacts: published.
- SM110 local installed artifact: correctness, compile, graph, and parity
  passed.
- SM110 Hub artifact: pending clean-commit rebuild and upload.

## NVIDIA Thor SM110 Results

Validated August 2, 2026 on NVIDIA Thor with PyTorch 2.11.0+cu130, CUDA 13.0,
CUTLASS 4.5.2, and a locally installed
`torch211-cxx11-cu130-aarch64-linux` artifact. Timings are CUDA Graph replay
latencies. `Native` is the independently loaded original FlashRT pointer API;
the ratio is installed artifact / native, so values above 1 are slower.

| Workload `(M,K,N)` | Auto tile | Artifact us | Native us | Artifact/native | Correctness |
| --- | --- | ---: | ---: | ---: | --- |
| decode `(1,4096,2048)` | T1 | 16.4 | 16.4 | 1.001 | pass |
| decode-wide `(1,4096,8192)` | T1 | 49.2 | 49.2 | 1.001 | pass |
| small-M `(16,4096,4096)` | T1 | 24.0 | 23.8 | 1.011 | pass |
| small-M `(32,4096,8192)` | T1 | 42.3 | 46.1 | 0.917 | pass |
| small-M `(64,512,1024)` | T1 | 6.5 | 6.4 | 1.003 | pass |
| PI0.5 QKV `(51,2048,2560)` | T1 | 12.3 | 12.3 | 1.000 | pass |
| PI0.5 O `(51,2048,2048)` | T1 | 10.5 | 10.5 | 1.008 | pass |
| PI0.5 gate/up `(51,2048,16384)` | T1 | 138.6 | 130.9 | 1.059 | pass |
| PI0.5 down `(51,8192,2048)` | T1 | 22.6 | 22.6 | 0.999 | pass |
| GROOT DiT QKV `(51,1536,4608)` | T1 | 14.4 | 14.4 | 1.000 | pass |
| GROOT N1.7 O `(277,2048,2048)` | Wide | 17.4 | 17.3 | 1.003 | pass |
| GROOT N1.7 gate/up `(277,2048,16384)` | Wide | 171.0 | 171.3 | 0.998 | pass |
| GROOT N1.7 down `(277,8192,2048)` | T1 | 58.4 | 56.3 | 1.037 | pass |
| GROOT vision O `(1024,1024,1024)` | Sq | 12.3 | 12.3 | 1.000 | pass |
| Cosmos Edge action `(64,2048,9216)` | T1 | 24.6 | 24.6 | 1.001 | pass |
| LingBot vision O `(1024,1280,1280)` | Wide | 16.4 | 16.4 | 1.000 | pass |
| LingBot action gate/up `(105,2048,16384)` | T1 | 143.4 | 145.4 | 0.986 | pass |

The stable PI0.5 gate/up row is 5.9% slower than the old native extension and
is retained explicitly rather than averaged away. It remains within the
internal per-path 10% release blocker, while source-to-artifact packaging
parity itself passed with p95/max 1.0068. No speedup claim is derived from rows
where dynamic Thor clocks made the artifact appear faster than native.
