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
| decode `(1,4096,2048)` | T1 | 17.232 | 17.344 | 0.997 | pass |
| decode-wide `(1,4096,8192)` | T1 | 50.144 | 50.112 | 1.001 | pass |
| small-M `(16,4096,4096)` | T1 | 23.952 | 24.304 | 0.997 | pass |
| small-M `(32,4096,8192)` | T1 | 46.256 | 46.000 | 1.013 | pass |
| small-M `(64,512,1024)` | T1 | 11.296 | 11.264 | 0.997 | pass |
| PI0.5 QKV `(51,2048,2560)` | T1 | 14.080 | 14.048 | 1.009 | pass |
| PI0.5 O `(51,2048,2048)` | T1 | 13.440 | 13.504 | 0.993 | pass |
| PI0.5 gate/up `(51,2048,16384)` | Wide | 92.496 | 81.312 | 1.128 | pass |
| PI0.5 down `(51,8192,2048)` | T1 | 23.392 | 23.392 | 1.003 | pass |
| GROOT DiT QKV `(51,1536,4608)` | T1 | 15.232 | 15.104 | 1.004 | pass |
| GROOT N1.7 O `(277,2048,2048)` | Wide | 18.704 | 18.704 | 1.002 | pass |
| GROOT N1.7 gate/up `(277,2048,16384)` | Wide | 186.432 | 189.360 | 0.985 | pass |
| GROOT N1.7 down `(277,8192,2048)` | Sq | 50.080 | 49.984 | 1.003 | pass |
| GROOT vision O `(1024,1024,1024)` | Sq | 15.360 | 15.424 | 0.997 | pass |
| Cosmos Edge action `(64,2048,9216)` | T1 | 25.264 | 25.312 | 0.999 | pass |
| LingBot vision O `(1024,1280,1280)` | Wide | 17.152 | 17.216 | 0.997 | pass |
| LingBot action gate/up `(105,2048,16384)` | Wide | 140.080 | 139.984 | 1.002 | pass |

Each graph ratio is the median of paired, per-launch package/native samples;
candidate order rotates every round to control Thor DVFS bias. Sixteen rows are
within about 1.3% of the original FlashRT pointer extension. PI0.5 gate/up is a
reproducible CUTLASS dependency-version outlier: the Hub-buildable package uses
CUTLASS 4.5.2 while the original native extension uses 4.4.2, and the paired
ratio in the final clean artifact run is 1.128. CUTLASS 4.0 was also tested but
failed at runtime on SM110, so it is not a valid packaging fallback. This row
is retained explicitly and is not used for a native-parity claim.

The installed artifact selected the fastest validated tactic on every row;
worst auto/fastest-valid-tile was 1.0028. Source-to-artifact packaging parity
passed with median 0.9986, p95 1.0195, and max 1.0244.
