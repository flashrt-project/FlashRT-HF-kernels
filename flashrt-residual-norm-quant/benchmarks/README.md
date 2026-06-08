# benchmarks

Package-level microbenchmarks for residual/RMSNorm/static-FP8 quantization.

Run:

```bash
python flashrt-residual-norm-quant/benchmarks/benchmark.py --backend source --shapes all
```

The benchmark reports FlashRT Tensor API latency against a PyTorch eager
reference for:

- `rms_norm_quant_fp8_static_bf16`
- `residual_add_rms_norm_quant_fp8_static_bf16`
