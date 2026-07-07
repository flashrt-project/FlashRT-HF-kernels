# Benchmark Results

Source-local RTX 5090 data from July 7, 2026. Built-artifact Hub data must be
refreshed after HF Jobs publishing before public performance claims.

Command:

```bash
python benchmarks/benchmark.py --backend source --mode headline
```

| Workload | M | K | N | Op | FlashRT source us | PyTorch eager us | Speedup |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| decode_m8 | 8 | 1024 | 2560 | `int8_rowwise_linear_bf16` | 6.196 | 24.525 | 3.96x |
| small_batch | 64 | 2048 | 8192 | `int8_rowwise_linear_bf16` | 8.603 | 92.356 | 10.74x |
| vision_prefill | 522 | 2048 | 2560 | `int8_rowwise_linear_bf16` | 16.584 | 149.920 | 9.04x |
| vision_prefill | 522 | 2048 | n/a | `rms_norm_quantize_int8_rowwise_bf16` | 6.186 | 67.741 | 10.95x |

Baseline is a PyTorch eager expression for the same INT8 math, not a full model
runtime comparison.
