# Benchmark Results: flashrt-fp8-ffn

## BF16 Region Entry: RTX 5090 Source RC (2026-07-17)

- Torch: `2.9.0a0+145a3a7bda.nv25.10`
- Warmup/iterations/rounds: `20/100/5`; primary FlashRT-vs-BF16 timing uses
  A-B-B-A ordering and reports the median samples.
- Baselines: allocation-free FlashRT BF16 entry, explicit CUDA Graph replay,
  old separate input quantization, FP8 kernel-only, BF16 PyTorch eager, and
  verified BF16 `torch.compile(fullgraph=True)`.
- Correctness: input quantization exact; all staged rows below have
  `max_abs=0`, `p99_abs=0`; FlashRT op compile and graph replay pass.

| Shape | FlashRT us | Graph us | Separate quant us | Kernel-only us | BF16 eager us | vs eager | BF16 compile us | vs separate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| siglip M8 | 18.491 | 16.395 | 35.107 | 16.445 | 28.717 | 1.55x | 37.895 | 1.90x |
| siglip M51 | 22.577 | 19.634 | 36.966 | 20.536 | 52.907 | 2.34x | 57.815 | 1.64x |
| siglip M64 | 22.579 | 20.496 | 36.550 | 20.542 | 71.414 | 3.16x | 58.674 | 1.62x |
| siglip M105 | 28.745 | 24.812 | 41.117 | 26.697 | 63.499 | 2.21x | 47.065 | 1.43x |
| siglip M128 | 28.742 | 22.538 | 41.118 | 27.386 | 65.514 | 2.28x | 47.365 | 1.43x |
| DiT M8 | 21.430 | 18.451 | 36.800 | 19.130 | 36.913 | 1.72x | 49.692 | 1.72x |
| DiT M51 | 24.628 | 21.566 | 37.039 | 22.576 | 47.568 | 1.93x | 66.141 | 1.50x |
| DiT M64 | 26.691 | 24.053 | 39.065 | 24.623 | 48.808 | 1.83x | 67.548 | 1.46x |
| DiT M105 | 34.886 | 28.679 | 49.283 | 34.513 | 60.481 | 1.73x | 63.454 | 1.41x |
| DiT M128 | 34.863 | 28.676 | 47.265 | 32.837 | 63.305 | 1.82x | 65.441 | 1.36x |

The BF16 baseline uses the original BF16 weights and activations; no FP8
dequantization is included in its timed region. Random per-tensor FP8 weight
quantization gives BF16-reference cosine `0.99851-0.99858`; package migration
parity is checked independently against the established FP8 staged path.
These are source-extension release-candidate results after vectorizing the
BF16 input producer, bias/GELU/FP8 producer, and final BF16 bias path. They are
not Hub built-artifact claims; the published artifact is benchmarked again
from a cold download before its numbers replace the built-artifact section.

## RTX 5090 Source-Extension Results

- Device: NVIDIA GeForce RTX 5090
- Compute capability: SM120
- Torch: 2.9.1+cu128
- Backend: local source extension
- Baselines: PyTorch eager reference and compile-stable `torch.compile`
  reference.
- `torch.compile` baseline status: the benchmark verifies compiled-reference
  output against eager output before reporting timing. The full FP8 FFN
  reference graph-breaks the numerically sensitive `GELU -> FP8 requant` and
  final BF16 bias/cast boundaries because a raw default-Inductor compile of the
  whole fake-quant chain is not bit-equivalent to eager on this stack.
- Precision gate: p99 absolute error <= 1.0 and p99 relative error with
  `abs(reference)` floored at 1.0 <= 0.05.

Commands:

```bash
python flashrt-fp8-ffn/tests/test_fp8_ffn.py --backend source
python flashrt-fp8-ffn/benchmarks/benchmark.py \
  --backend source \
  --shapes all \
  --warmup 3 \
  --iters 10 \
  --output internal-tests/flashrt-fp8-ffn/benchmark-source-rtx5090-expanded-all.json \
  --markdown internal-tests/flashrt-fp8-ffn/benchmark-source-rtx5090-expanded-all.md
python flashrt-fp8-ffn/benchmarks/benchmark.py \
  --backend source \
  --shapes headline \
  --compile-baseline \
  --warmup 5 \
  --iters 20 \
  --output internal-tests/flashrt-fp8-ffn/benchmark-source-rtx5090-headline-compile.json \
  --markdown internal-tests/flashrt-fp8-ffn/benchmark-source-rtx5090-headline-compile.md
```

## Headline Rows

These rows compare the full FP8 GELU MLP sublayer against the PyTorch eager
reference and the compile-stable `torch.compile` reference. The eager reference
is the correctness baseline; compiled-reference output must match eager before
timing is reported.

| Shape | M,K,H,N | Layers | FlashRT us | Eager us | vs eager | Compile us | vs compile | Compile status | P99 abs | P99 rel | Max abs | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| pi05_decoder_ffn_m10 | 10,1024,4096,1024 | 18 | 368.298 | 2435.629 | 6.61x | 2247.846 | 6.10x | ok | 0.0000 | 0.000000 | 0.0625 | PASS |
| pi05_vision_ffn_2view | 512,1152,4304,1152 | 27 | 1671.968 | 10852.866 | 6.49x | 10099.493 | 6.04x | ok | 0.0000 | 0.000000 | 32.0000 | PASS |
| groot_vit_ffn_2view | 512,1024,4096,1024 | 24 | 1179.680 | 8476.130 | 7.19x | 7859.718 | 6.66x | ok | 0.0000 | 0.000000 | 16.0000 | PASS |
| groot_vl_self_attn_ffn_seq1024 | 1024,2048,8192,2048 | 4 | 981.026 | 6387.746 | 6.51x | 5782.505 | 5.89x | ok | 0.0000 | 0.000000 | 64.0000 | PASS |

### torch.compile Baseline Note

The package-local benchmark validates compiled-reference output before timing
it. A raw default-Inductor compile of this full FP8 FFN fake-quant chain is not
bit-equivalent to eager: the `GELU -> FP8 requant` boundary can move values
across FP8 rounding thresholds, and the second GEMM amplifies the difference.
The reported compile baseline is therefore a segmented compile-stable reference:
it keeps the FP8 dequant GEMMs in compiled regions and graph-breaks the
requantization and final BF16 bias/cast boundaries.

## Full Shape Sweep

These rows cover the first-version PI0.5/GROOT FFN shape grid. Compile
baselines are intentionally limited to the headline rows to keep the pre-build
development loop tractable.

| Shape | M,K,H,N | Layers | FlashRT us | Eager us | vs eager | P99 abs | P99 rel | Max abs | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pi05_decoder_ffn_m1 | 1,1024,4096,1024 | 18 | 386.349 | 2113.206 | 5.47x | 0.0000 | 0.000000 | 0.0000 | PASS |
| pi05_decoder_ffn_m8 | 8,1024,4096,1024 | 18 | 368.653 | 2684.435 | 7.28x | 0.0000 | 0.000000 | 1.0000 | PASS |
| pi05_decoder_ffn_m10 | 10,1024,4096,1024 | 18 | 367.421 | 2669.581 | 7.27x | 0.0000 | 0.000000 | 2.0000 | PASS |
| pi05_decoder_ffn_m16 | 16,1024,4096,1024 | 18 | 406.147 | 2993.133 | 7.37x | 0.0000 | 0.000000 | 8.0000 | PASS |
| pi05_vision_ffn_1view | 256,1152,4304,1152 | 27 | 1230.659 | 7893.139 | 6.41x | 0.0000 | 0.000000 | 32.0000 | PASS |
| pi05_vision_ffn_2view | 512,1152,4304,1152 | 27 | 1659.629 | 10635.197 | 6.41x | 0.0000 | 0.000000 | 32.0000 | PASS |
| pi05_vision_ffn_3view | 768,1152,4304,1152 | 27 | 2023.162 | 13351.164 | 6.60x | 0.0000 | 0.000000 | 16.0000 | PASS |
| groot_vit_ffn_1view | 256,1024,4096,1024 | 24 | 740.294 | 7113.017 | 9.61x | 0.0000 | 0.000000 | 16.0000 | PASS |
| groot_vit_ffn_2view | 512,1024,4096,1024 | 24 | 1180.022 | 8447.917 | 7.16x | 0.0000 | 0.000000 | 32.0000 | PASS |
| groot_vit_ffn_4view | 1024,1024,4096,1024 | 24 | 1867.638 | 11957.926 | 6.40x | 0.0000 | 0.000000 | 32.0000 | PASS |
| groot_deepstack_merge_2view | 128,4096,4096,2048 | 3 | 107.382 | 1231.066 | 11.46x | 0.0000 | 0.000000 | 32.0000 | PASS |
| groot_vl_self_attn_ffn_seq512 | 512,2048,8192,2048 | 4 | 492.291 | 3809.549 | 7.74x | 0.0000 | 0.000000 | 64.0000 | PASS |
| groot_vl_self_attn_ffn_seq1024 | 1024,2048,8192,2048 | 4 | 980.454 | 6326.080 | 6.45x | 0.0000 | 0.000000 | 64.0000 | PASS |
| groot_vl_self_attn_ffn_seq2520 | 2520,2048,8192,2048 | 4 | 1932.403 | 14522.385 | 7.52x | 0.7500 | 0.004274 | 64.0000 | PASS |
| groot_action_dit_ffn | 41,1536,6144,1536 | 32 | 905.354 | 8060.080 | 8.90x | 0.0000 | 0.000000 | 32.0000 | PASS |

## Interpretation

This package measures complete FP8 GELU MLP sublayers:

```text
FP8 up GEMM -> bias/GELU -> FP8 quant -> FP8 down GEMM -> bias
```

It is a stronger first-version showcase surface than epilogue-only fragments
because it includes both FP8 GEMMs and the activation/quantization bridge. The
first-version implementation uses the cuBLASLt FP8 path with row-major tensor
APIs. It has not yet been replaced by shape-locked CUTLASS/megakernel kernels,
so these numbers should be read as strong reusable package results, not a
proof that the FlashRT production serving path has reached its final optimal
tile for every shape.

## RTX 5090 Built-Artifact Results

- Device: NVIDIA GeForce RTX 5090
- Compute capability: SM120
- Driver: 580.82.07
- Variant: `torch211-cxx11-cu128-x86_64-linux`
- Built from commit: `21417e6`
- Torch inside HF testshell: 2.11.0+cu128
- Backend: copied `kernel-builder` artifact
- Precision gate: same as source-extension results.

Commands:

```bash
python flashrt-fp8-ffn/tests/test_fp8_ffn.py \
  --backend installed \
  --artifact flashrt-fp8-ffn/build/torch211-cxx11-cu128-x86_64-linux
python flashrt-fp8-ffn/benchmarks/benchmark.py \
  --backend installed \
  --artifact flashrt-fp8-ffn/build/torch211-cxx11-cu128-x86_64-linux \
  --shapes all \
  --warmup 3 \
  --iters 10
python flashrt-fp8-ffn/benchmarks/benchmark.py \
  --backend installed \
  --artifact flashrt-fp8-ffn/build/torch211-cxx11-cu128-x86_64-linux \
  --shapes headline \
  --compile-baseline \
  --warmup 5 \
  --iters 20
```

### Built-Artifact Headline Rows

| Shape | M,K,H,N | Layers | FlashRT us | Eager us | vs eager | Compile us | vs compile | Compile status | P99 abs | P99 rel | Max abs | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| pi05_decoder_ffn_m10 | 10,1024,4096,1024 | 18 | 368.298 | 2435.629 | 6.61x | 2247.846 | 6.10x | ok | 0.0000 | 0.000000 | 0.0625 | PASS |
| pi05_vision_ffn_2view | 512,1152,4304,1152 | 27 | 1671.968 | 10852.866 | 6.49x | 10099.493 | 6.04x | ok | 0.0000 | 0.000000 | 32.0000 | PASS |
| groot_vit_ffn_2view | 512,1024,4096,1024 | 24 | 1179.680 | 8476.130 | 7.19x | 7859.718 | 6.66x | ok | 0.0000 | 0.000000 | 16.0000 | PASS |
| groot_vl_self_attn_ffn_seq1024 | 1024,2048,8192,2048 | 4 | 981.026 | 6387.746 | 6.51x | 5782.505 | 5.89x | ok | 0.0000 | 0.000000 | 64.0000 | PASS |

### Built-Artifact Full Shape Sweep

| Shape | M,K,H,N | Layers | FlashRT us | Eager us | vs eager | P99 abs | P99 rel | Max abs | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pi05_decoder_ffn_m1 | 1,1024,4096,1024 | 18 | 367.178 | 1917.309 | 5.22x | 0.0000 | 0.000000 | 0.0000 | PASS |
| pi05_decoder_ffn_m8 | 8,1024,4096,1024 | 18 | 368.029 | 2407.398 | 6.54x | 0.0000 | 0.000000 | 1.0000 | PASS |
| pi05_decoder_ffn_m10 | 10,1024,4096,1024 | 18 | 367.606 | 2434.016 | 6.62x | 0.0000 | 0.000000 | 2.0000 | PASS |
| pi05_decoder_ffn_m16 | 16,1024,4096,1024 | 18 | 405.894 | 2652.019 | 6.53x | 0.0000 | 0.000000 | 8.0000 | PASS |
| pi05_vision_ffn_1view | 256,1152,4304,1152 | 27 | 1097.354 | 7670.819 | 6.99x | 0.0000 | 0.000000 | 32.0000 | PASS |
| pi05_vision_ffn_2view | 512,1152,4304,1152 | 27 | 1653.162 | 10703.955 | 6.47x | 0.0000 | 0.000000 | 32.0000 | PASS |
| pi05_vision_ffn_3view | 768,1152,4304,1152 | 27 | 2028.134 | 13226.682 | 6.52x | 0.0000 | 0.000000 | 16.0000 | PASS |
| groot_vit_ffn_1view | 256,1024,4096,1024 | 24 | 738.490 | 7019.769 | 9.51x | 0.0000 | 0.000000 | 16.0000 | PASS |
| groot_vit_ffn_2view | 512,1024,4096,1024 | 24 | 1165.082 | 8427.415 | 7.23x | 0.0000 | 0.000000 | 32.0000 | PASS |
| groot_vit_ffn_4view | 1024,1024,4096,1024 | 24 | 1874.026 | 12089.379 | 6.45x | 0.0000 | 0.000000 | 32.0000 | PASS |
| groot_deepstack_merge_2view | 128,4096,4096,2048 | 3 | 110.906 | 1266.422 | 11.42x | 0.0000 | 0.000000 | 32.0000 | PASS |
| groot_vl_self_attn_ffn_seq512 | 512,2048,8192,2048 | 4 | 501.894 | 3846.003 | 7.66x | 0.0000 | 0.000000 | 64.0000 | PASS |
| groot_vl_self_attn_ffn_seq1024 | 1024,2048,8192,2048 | 4 | 949.357 | 6317.344 | 6.65x | 0.0000 | 0.000000 | 64.0000 | PASS |
| groot_vl_self_attn_ffn_seq2520 | 2520,2048,8192,2048 | 4 | 1932.896 | 14478.560 | 7.49x | 0.7500 | 0.004274 | 64.0000 | PASS |
| groot_action_dit_ffn | 41,1536,6144,1536 | 32 | 900.989 | 8013.347 | 8.89x | 0.0000 | 0.000000 | 32.0000 | PASS |

Multi-hardware results are still pending.
