# Benchmarks

```bash
python fp8-gemm/benchmarks/benchmark.py --backend source --mode headline
```

The benchmark sweeps dispatcher rows and explicit M=1 GEMV variants.

On NVIDIA Thor, use `--mode thor-full`. The runner records normal launches,
CUDA Graph replay, all SM110 Sq/T1/Wide diagnostic tiles, and an optional
original FlashRT pointer-API comparison when `FLASHRT_NATIVE_ROOT` is set.
