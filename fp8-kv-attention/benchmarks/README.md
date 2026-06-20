# Benchmarks

Run:

```bash
python fp8-kv-attention/benchmarks/benchmark.py --backend source --mode full
```

Benchmark results should be reported against a PyTorch reference that dequants
the same FP8 cache and uses the same causal/speculative mask.
