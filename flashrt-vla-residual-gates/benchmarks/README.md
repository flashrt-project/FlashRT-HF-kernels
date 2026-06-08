# benchmarks

Package-level microbenchmarks for VLA joint residual/gate APIs.

Run:

```bash
python flashrt-vla-residual-gates/benchmarks/benchmark.py --backend source --shapes all
```

The benchmark compares the fused FlashRT Tensor API against a PyTorch eager
reference chain that performs the same video/action/und residual updates.
