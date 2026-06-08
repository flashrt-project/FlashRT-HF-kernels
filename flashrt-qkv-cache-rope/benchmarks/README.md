# benchmarks

Package-level microbenchmarks for packed-QKV split, Q/K RMSNorm, and RoPE.

Run:

```bash
python flashrt-qkv-cache-rope/benchmarks/benchmark.py --backend source --shapes all
```

The benchmark compares the fused FlashRT Tensor API against a PyTorch eager
reference chain that performs split, RMSNorm Q/K, and RoPE Q/K.
