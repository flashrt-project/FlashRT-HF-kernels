# benchmarks

Package-level microbenchmarks for spatiotemporal BF16 layout/cache APIs.

Run:

```bash
python flashrt-spatiotemporal-layout/benchmarks/benchmark.py --backend source --shapes all
```

The benchmark compares each FlashRT Tensor API against the equivalent PyTorch
eager layout/reference operation.
