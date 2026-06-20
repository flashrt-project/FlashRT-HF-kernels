# fp4-gemm Benchmark Results

Source benchmark on NVIDIA GeForce RTX 5090, PyTorch `2.9.1+cu128`.

Command:

```bash
python fp4-gemm/benchmarks/benchmark.py \
  --mode headline \
  --warmup 100 \
  --iterations 500 \
  --json-out internal-tests/fp4-gemm-source-benchmark-rerun-long.json
```

Reference is PyTorch GEMM over the same dequantized FP4/SFA and FP4/SFB inputs
that the FlashRT kernel consumes.

| Shape | Variant | FlashRT us | Torch reference us | Speedup | Max abs | P99 abs | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M=16, N=128, K=128 | 0 | 6.014 | 16.159 | 2.69x | 0.0 | 0.0 | 1.0 |
| M=16, N=128, K=128 | 1 | 6.239 | 16.159 | 2.59x | 0.0 | 0.0 | 1.0 |
| M=16, N=128, K=128 | 2 | 6.043 | 16.159 | 2.67x | 0.0 | 0.0 | 1.0 |
| M=32, N=256, K=256 | 0 | 6.116 | 18.042 | 2.95x | 0.0 | 0.0 | 1.0 |
| M=32, N=256, K=256 | 1 | 8.204 | 18.042 | 2.20x | 0.0 | 0.0 | 1.0 |
| M=32, N=256, K=256 | 2 | 6.112 | 18.042 | 2.95x | 0.0 | 0.0 | 1.0 |
| M=64, N=512, K=512 | 0 | 5.995 | 17.905 | 2.99x | 0.0 | 0.0 | 1.0 |
| M=64, N=512, K=512 | 1 | 9.429 | 17.905 | 1.90x | 0.0 | 0.0 | 1.0 |
| M=64, N=512, K=512 | 2 | 6.366 | 17.905 | 2.81x | 0.0 | 0.0 | 1.0 |

Variant notes:

- `variant=0` is the stable default.
- `variant=1` is the widen schedule intended for very large `N`; it is not the
  best choice for these small validation shapes.
- `variant=2` is competitive on small shapes and remains exposed for explicit
  A/B testing.

Installed-artifact benchmark rows should be regenerated after HF Jobs publishes
the package.
