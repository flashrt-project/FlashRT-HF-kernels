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

## BF16 Direct Producer

Source benchmark on RTX 5090 with 100 warmup and 1000 measured iterations:

| Shape | Direct BF16 us | Cast + FP16 producer us | Speedup | Native BF16 us | Wrapper/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| M=1, K=5120 | 4.098 | 6.404 | 1.563x | 6.150 | 0.666x |
| M=1, K=6144 | 4.098 | 6.403 | 1.562x | 8.190 | 0.500x |
| M=1, K=17408 | 4.096 | 6.413 | 1.566x | 18.442 | 0.222x |

The direct entry is byte-exact against the package's established
BF16-to-FP16 plus FP16-producer contract. The native timing is reported as a
performance reference only because that producer uses a distinct quantization
strategy.

## NVIDIA Thor SM110 Results (installed artifact, 2026-08-04)

Measured against `/data/test_thor/fp4-gemm` (installed artifact) on Thor
`sm_110a`, torch `2.9.1+cu130`. `nvfp4_gemm_bf16` routes SM110 through the
dedicated CUTLASS `sm110_gemm_dispatch` path (not the SIMT fallback), so the
production GEMM is fast on Thor. Reference = PyTorch fp32 GEMM over the same
dequantized FP4/SFA+FP4/SFB inputs.

| Workload `(M,N,K)` | FlashRT us | Eager us | Compile us | vs eager | Max abs | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pi0.5 action gate/up `(51,16384,2048)` | 38.710 | 2400.608 | 2117.904 | 62.0x | 0.0 | 1.000000 |
| pi0.5 action down `(51,2048,8192)` | 18.074 | 1330.918 | 1268.806 | 73.6x | 0.0 | 1.000000 |
| GROOT DiT QKV `(51,4608,1536)` | 12.752 | 545.200 | 500.822 | 42.8x | 0.0 | 1.000000 |
| GROOT backbone gate/up `(277,16384,2048)` | 79.242 | 7006.390 | 6748.576 | 88.4x | 0.0 | 1.000000 |
| Cosmos Edge action `(64,9216,2048)` | 21.398 | 1446.109 | 1312.918 | 67.6x | 0.0 | 1.000000 |
| LingBot action gate/up `(105,16384,2048)` | 31.331 | 2947.763 | 2856.307 | 94.1x | 0.0 | 1.000000 |

The SM110 CUTLASS GEMM is 42-94x faster than the dequantized PyTorch reference
with bitwise-equal output. The portable SIMT fallbacks (`nvfp4_gemm_linear_simt`
et al.) remain the compatibility path for fused-epilogue ops without an SM110
CUTLASS implementation and are exercised by the on-device correctness suite.
