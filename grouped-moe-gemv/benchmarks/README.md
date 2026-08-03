# Benchmarks

Run on the target GPU after correctness passes:

```bash
python grouped-moe-gemv/benchmarks/benchmark.py --backend source
```

The harness compares the complete W4A4 producer+grouped region against the
native W4A16 grouped path, a per-route grouped loop, and the legacy per-pair
quantize+GEMV loop. It also reports W4A4 kernel-only latency so an upstream
fused FP4 producer can be evaluated without charging a duplicate producer.
