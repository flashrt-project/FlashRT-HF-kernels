# Benchmarks

Benchmark with preallocated output, scratch and barrier tensors. Report eager,
compiled, original FlashRT and packaged-region latency under identical scales.

```bash
python benchmarks/benchmark.py --backend source
python benchmarks/benchmark.py --backend installed --artifact /path/to/variant
```
