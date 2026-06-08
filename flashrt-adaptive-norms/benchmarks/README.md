# benchmarks

Package-level microbenchmarks for adaptive norm APIs.

Run:

```bash
python flashrt-adaptive-norms/benchmarks/benchmark.py --backend source --shapes all
```

The benchmark compares the fused FlashRT Tensor API against a PyTorch eager
reference chain that performs the same adaptive norm and FP8 quantization math.
