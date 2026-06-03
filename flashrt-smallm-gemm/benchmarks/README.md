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

## Comparison Stack

Public results for this package should follow
`../../docs/kernel-comparison-matrix.md`.

- Decode `M=1` W4A4 matvec compares against PyTorch dequant+matmul,
  compiled dequant+matmul, and a strong low-bit baseline when available.
- Strong low-bit baselines may be CUTLASS, cuBLASLt, or an existing FlashRT
  internal production path with the same math and layout contract.
- Rows that are correct but slower than a strong baseline are labeled
  `compatibility` or `reject`; they are not package headline rows.
- Dispatch boundaries must be benchmark-backed across decode and small-M grids
  before exposing a generic dispatcher.
