# Benchmarks

Implemented benchmark groups:

- Layout conversion latency and bandwidth for scale-factor tensors.

Planned benchmark groups:

- Fused `GEMM + bias + GELU + FP4 quant` against unfused CUTLASS/cuBLAS plus
  separate epilogue kernels.
- Stream-K down GEMM against the strongest available CUTLASS/cuBLASLt path for
  the same shape and dtype.
