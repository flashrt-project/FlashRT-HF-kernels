# Benchmarks

Public result ledger:

```text
RESULTS.md
```

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

## Comparison Stack

Public results for this package should follow
`../../docs/kernel-comparison-matrix.md`.

- Scale-factor layout helpers compare against byte-parity references and
  tensor-layout PyTorch references; report latency and bandwidth.
- `torch.compile` is reported only when the layout reference is tensor-only and
  can compile without Python loops or CPU copies.
- NVFP4/FP4 GEMM epilogues require CUTLASS/cuBLASLt or an unfused CUDA GEMM plus
  separate epilogue baseline before headline claims.
- Keep SM120 paths labeled CUDA 12.8+ SM120 until the package includes and
  validates a broader CUDA architecture source path.
