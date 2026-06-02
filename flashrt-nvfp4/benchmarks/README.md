# Benchmarks

Implemented benchmark groups:

- Layout conversion latency and bandwidth for scale-factor tensors.

Current layout benchmark grid:

- row-boundary cases with `D=4096`: rows `1,2,31,32,33,127,128,129`;
- contracted-dimension cases with `rows=16`: `D=1024,2048,8192,12288`;
- VLA/video wide case: `(rows,D)=(64,16384)`.

Planned benchmark groups:

- Fused `GEMM + bias + GELU + FP4 quant` against unfused CUTLASS/cuBLAS plus
  separate epilogue kernels.
- Stream-K down GEMM against the strongest available CUTLASS/cuBLASLt path for
  the same shape and dtype.
