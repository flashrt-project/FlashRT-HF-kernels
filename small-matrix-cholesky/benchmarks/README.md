# Benchmarks

`benchmark.py` measures preallocated package output against the same
preallocated `torch.linalg.cholesky_ex` reference in warmed eager and
`torch.compile(fullgraph=True)` modes for the three equal-input-footprint batch
shapes used during development:

- `(batch, n) = (4096, 32)`
- `(batch, n) = (1024, 64)`
- `(batch, n) = (256, 128)`

The script reports median latency, speedup, approximate Cholesky TFLOP/s, and
input-plus-output bandwidth. Allocation and result memoization are excluded.
