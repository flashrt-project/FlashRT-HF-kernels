# Benchmarks

Run:

```bash
python benchmarks/benchmark.py --backend source --warmup 50 --iters 500
```

The PyTorch eager baseline includes the explicit cache concat, layout
conversion, conv3d, bias, BF16 rounding, and residual add.
