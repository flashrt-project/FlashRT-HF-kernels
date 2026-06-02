# Benchmarks

Planned benchmark groups:

- Decode `M=1` W4A4 matvec vs CUTLASS/cuBLASLt and PyTorch dequant+matmul.
  First shape grid: `K in {4096, 12288}` and `N in {1024, 4096, 12288}`.
- Small-M W4A4 warpsplit vs generic CUTLASS low-bit GEMM where available.
- Tiny FP8 fixed-family kernels vs cuBLASLt FP8 and PyTorch reference chains.
- Shape sweep must include out-of-grid cases to justify dispatch boundaries.
