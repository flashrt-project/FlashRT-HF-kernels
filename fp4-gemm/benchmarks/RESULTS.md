# fp4-gemm Benchmark Results

Installed kernel-builder artifact benchmark on NVIDIA GeForce RTX 5090,
PyTorch `2.11.0+cu128`.

Command:

```bash
python fp4-gemm/benchmarks/benchmark.py \
  --backend installed \
  --artifact fp4-gemm/build/torch211-cxx11-cu128-x86_64-linux \
  --mode headline \
  --warmup 100 \
  --iterations 500 \
  --json-out internal-tests/fp4-gemm-installed-benchmark.json
```

Reference is PyTorch GEMM over the same dequantized FP4/SFA and FP4/SFB inputs
that the FlashRT kernel consumes.

| Shape | Variant | FlashRT us | Eager us | Compile us | vs eager | vs compile | Max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M=16, N=128, K=128 | 0 | 6.156 | 15.156 | 27.748 | 2.46x | 4.51x | 0.0 |
| M=16, N=128, K=128 | 1 | 6.152 | 15.156 | 27.748 | 2.46x | 4.51x | 0.0 |
| M=16, N=128, K=128 | 2 | 6.145 | 15.156 | 27.748 | 2.47x | 4.52x | 0.0 |
| M=32, N=256, K=256 | 0 | 6.153 | 16.685 | 35.690 | 2.71x | 5.80x | 0.0 |
| M=32, N=256, K=256 | 1 | 8.201 | 16.685 | 35.690 | 2.03x | 4.35x | 0.0 |
| M=32, N=256, K=256 | 2 | 6.147 | 16.685 | 35.690 | 2.71x | 5.81x | 0.0 |
| M=64, N=512, K=512 | 0 | 6.152 | 16.480 | 36.205 | 2.68x | 5.89x | 0.0 |
| M=64, N=512, K=512 | 1 | 10.246 | 16.480 | 36.205 | 1.61x | 3.53x | 0.0 |
| M=64, N=512, K=512 | 2 | 6.152 | 16.480 | 36.205 | 2.68x | 5.89x | 0.0 |

Variant notes:

- `variant=0` is the stable default.
- `variant=1` is the widen schedule intended for very large `N`; it is not the
  best choice for these small validation shapes.
- `variant=2` is competitive on small shapes and remains exposed for explicit
  A/B testing.

The PyTorch references consume the same already-dequantized FP4 tensors and do
not include quantization. The compiled reference is warmed before timing.
