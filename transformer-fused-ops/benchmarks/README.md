# Benchmarks

Benchmark after correctness passes.

Run the SM110 BF16 producer benchmark with:

```bash
python transformer-fused-ops/benchmarks/benchmark_bf16_producers.py \
  --backend source
```

It compares fused static-FP8 quantize, LayerNorm producer, and merged GeGLU
producer entries against mathematically equivalent PyTorch eager graphs. The
correctness gate requires FP8 p99 absolute error `0` and cosine at least
`0.9999`; the full package test additionally covers tails, compile, and graph
replay.
