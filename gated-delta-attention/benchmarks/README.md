# Benchmarks

Built-artifact benchmarks should be regenerated after HF Jobs upload.

H32/H16 source or installed-artifact benchmark:

```bash
python gated-delta-attention/benchmarks/benchmark_h32_pipeline.py \
  --backend source --sequences 1,4,64
```
