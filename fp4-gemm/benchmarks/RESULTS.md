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

## PI0.5 Thor Batch 3 BF16 schedules

Source release candidate on NVIDIA Jetson AGX Thor, PyTorch `2.13.0+cu130`,
CUDA 13.0. The package BF16 Tensor API is compared with the corresponding
FlashRT native FP16 v7/v10 or FP4-output launcher using preallocated buffers,
CUDA Graph replay, A-B-B-A ordering, and the minimum of repeated runs. This is
a schedule/parity gate across two input dtypes, not a claim that BF16 and FP16
have identical arithmetic contracts.

| Shape/family | Tile | Hub us | Native us | Hub/native |
| --- | ---: | ---: | ---: | ---: |
| decoder QKV `(10,2560,1024)` | v10 | 10.286 | 10.279 | 1.001x |
| decoder O `(10,1024,2048)` | v10 | 10.222 | 9.907 | 1.032x |
| decoder gate/up `(10,8192,1024)` | v10 | 20.515 | 20.516 | 1.000x |
| decoder down `(10,1024,4096)` | v10 | 11.333 | 11.308 | 1.002x |
| encoder gate/up `(576,16384,2048)` | v7 | 283.417 | 276.545 | 1.025x |
| encoder gate/up `(970,16384,2048)` | v7 | 376.212 | 373.363 | 1.008x |
| encoder down `(576,2048,16384)` | v7 | 170.183 | 167.915 | 1.014x |
| SigLIP `(512,4304,1152)` | v7 | 26.785 | 30.475 | 0.879x |
| SigLIP `(768,4304,1152)` | v7 | 34.984 | 34.996 | 1.000x |
| decoder O FP4-output `(10,1024,2048)` | native | 10.244 | 10.262 | 0.998x |

The strict source gate passed `72/72` and covers both ends of the decoder
`M=10..64` and encoder `M=576..970` bands. All BF16-output GEMM shapes are
checked against GEMM over the same dequantized inputs. FP4-output reached
cosine `0.995382`; the
bind-time MSE packer reduced reconstruction MSE from `0.000570887` to
`0.000449985`. BF16 v7/v10 and FP4-output CUDA Graph replay were bit-identical.

## SigLIP logical-width packaging gate

SigLIP's logical hidden width `4304` is zero-padded once at bind time to the
physical NVFP4 width `4320`. Runtime GEMM then consumes static packed tensors;
there is no hot-path padding or allocation. The public helper contract covers
both weight layouts `(4304, K)` and `(N, 4304)`, pads bias consistently, and
returns the logical dimensions for final output slicing.

On NVIDIA Thor with PyTorch `2.13.0+cu130`, the source release gate passed
`73/73`, including physical SigLIP up/down shapes, BF16 MSE packing, CUDA Graph
replay, and unsupported-shape rejection. The BF16 MSE packer reduced
reconstruction MSE from `0.000571271` to `0.000450529`. The RTX 5090 source
regression passed `26/26`. Installed-artifact validation is required after the
Hub build is published.
