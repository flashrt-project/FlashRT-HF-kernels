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

Result: 8/8 checks passed.

Covered public v1 rows:

- M=1 decode GEMV: `K in {512,4096}`, `N in {512,2048,8192}`
- small-M GEMM: `M in {8,16,32,64}` with representative
  transformer/diffuser-adjacent `K,N` rows
- M=1 residual-add GEMV

Metrics:

- `max_abs`
- `mean_abs`
- `p99_abs`
- cosine similarity
- output dtype
- tolerance

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

## Pending Release Gates

- `kernel-builder check-config` was not run in the current shell because `nix`
  is not installed there.
- HF Jobs build/upload is pending.
- Installed-artifact correctness through `get_kernel("flashrt/fp8-gemm")` is
  pending after upload.
