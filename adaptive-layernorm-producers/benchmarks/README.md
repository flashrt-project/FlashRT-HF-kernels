# Benchmarks

Run source benchmarks:

```bash
python adaptive-layernorm-producers/benchmarks/benchmark.py --backend source --iters 100
```

Run installed-artifact benchmarks after Hub packaging:

```bash
python adaptive-layernorm-producers/benchmarks/benchmark.py --backend installed --iters 100
```

`RESULTS.md` records the current local source-build numbers. Public benchmark
claims should be refreshed after installed-artifact testing on the target
hardware.
