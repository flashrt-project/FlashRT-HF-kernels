# fp4-gemm Benchmark Results

## SM110 Bias Epilogue Dispatch (2026-08-07)

Source RC on NVIDIA Thor, Torch `2.11.0+cu130`, CUDA 13.0. Timings are from
the final public APIs with preallocated outputs, 20 warmup iterations, 100
measured iterations, and the median of five CUDA-event rounds. The prior
column is the former fixed `128x64x256` tile measured by the same harness.

| Shape `(M,N,K)` | Family | Prior us | Tuned us | Speedup |
| --- | --- | ---: | ---: | ---: |
| `(41,6144,1536)` | bias | 20.727 | 15.540 | 1.334x |
| `(41,6144,1536)` | bias+residual | 22.668 | 16.519 | 1.372x |
| `(41,6144,1536)` | bias+GELU->FP4 | 22.648 | 16.533 | 1.370x |
| `(41,1536,6144)` | bias | 20.620 | 14.477 | 1.424x |
| `(41,1536,6144)` | bias+residual | 20.650 | 15.249 | 1.354x |
| `(41,1536,6144)` | bias+GELU->FP4 | 20.900 | 16.084 | 1.299x |

The final dispatch uses the `K=128` tile for the small-K residual path and the
short-M expanding GELU path; other rows use the `K=256` tile. Correctness and
model-shape acceptance are documented separately in `VALIDATION.md`.

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

## NVIDIA Thor GROOT N1.7 artifact

The SM110 additions from FlashRT
`24df793f4fa2d50780aea03b644208c6e0cb4162` were rebuilt on NVIDIA Thor with
PyTorch 2.13.0+cu130 as `torch213-cxx11-cu130-aarch64-linux`. The installed
artifact passed 23/23 checks; BF16-to-FP4 output was exact and the fullgraph
compile path had `max_abs=0`.

The FP4 quantizer Tensor wrapper/raw registered-op measurement was
`5.5812/5.0712 us` in direct mode. This eager delta includes Python-side
allocation and dispatch. With caller-owned buffers under CUDA Graph, the
measurement was `3.2988/3.3003 us` (`0.9995x`), which is the production GROOT
hot-path contract.
