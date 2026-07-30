# Validation

Date: June 20, 2026

Local environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8
- Source build target: `sm_120a`

## Source Correctness

Command:

```bash
python fp8-gemm/tests/test_fp8_gemm.py --backend source --mode full
```

Result: 14/14 checks passed, plus the blockwise custom op passed
`torch.compile(fullgraph=True)` with bitwise-equal output to the eager wrapper.

Covered public v1 rows:

- M=1 decode GEMV: `K in {512,4096}`, `N in {512,2048,8192}`
- small-M GEMM: `M in {8,16,32,64}` with representative
  transformer/diffuser-adjacent `K,N` rows
- M=1 residual-add GEMV
- block-128 scaled FP8 GEMM at:
  - `(M,K,N)=(1,1024,1024)`
  - `(51,1536,1536)`
  - `(277,2048,2048)`
  - `(1024,1152,1152)`
  - `(2520,3072,3072)`
  - `(128,4096,12288)`

Metrics:

- `max_abs`
- `mean_abs`
- `p99_abs`
- cosine similarity
- output dtype
- tolerance

The blockwise rows use the stricter gate:

- `max_abs <= 0.0625`
- `mean_abs <= 0.003`
- `p99_abs <= 0.015625`
- cosine similarity `>= 0.9999`

The release benchmark also compares the Tensor wrapper against an independent
binding of the original FlashRT pointer API. Matching source code alone is not
treated as proof of zero wrapper overhead.

## Source Benchmark

Command:

```bash
python fp8-gemm/benchmarks/benchmark.py \
  --backend source --mode headline --warmup 20 --iterations 100 --compile-ref
```

Result: all public rows passed. Headline rows are recorded in
`benchmarks/RESULTS.md`.

## Current Scope Boundary

Public v1 supports `M=1` and `2 <= M <= 64`.

M=128 is intentionally not exposed in v1. The correct tile tested locally did
not meet the performance bar, and an alternate big tile returned non-zero
status for the tested public row. This remains an internal tuning item rather
than a public API.

## HF Jobs Publish Status

`flashrt/fp8-gemm` v1 was built and uploaded through the repository HF Jobs
workflow.

- Hub revision checked on June 20, 2026: `166f09be`
- Uploaded variants:
  - `torch211-cxx11-cu128-x86_64-linux`
  - `torch211-cxx11-cu130-x86_64-linux`
  - `torch212-cxx11-cu130-x86_64-linux`
  - `torch212-cxx11-cu132-x86_64-linux`

Installed-artifact correctness through `get_kernel("flashrt/fp8-gemm")`
should be rerun in a torch211 or torch212 CUDA environment. The local
development environment used for the source tests is PyTorch 2.9.1+cu128,
which intentionally does not match the uploaded torch211/torch212 variants.
