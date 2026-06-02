# Benchmarks

Public result ledger:

```text
RESULTS.md
```

Current benchmark scope:

- Decode `M=1` W4A4 matvec vs CUTLASS/cuBLASLt and PyTorch dequant+matmul.
  First shape grid: `K in {4096, 12288}` and `N in {1024, 4096, 12288}`.

The public HF-style benchmark script currently covers the first decode shape
grid and verifies deterministic constant-input output:

```text
benchmark_nvfp4_w4a4_decode_matvec.py
```

Queued benchmark groups for later source slices:

- Small-M W4A4 warpsplit vs generic CUTLASS low-bit GEMM where available.
- Tiny FP8 fixed-family kernels vs cuBLASLt FP8 and PyTorch reference chains.
- Shape sweep must include out-of-grid cases to justify dispatch boundaries.
