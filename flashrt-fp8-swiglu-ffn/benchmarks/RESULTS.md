# Benchmark Results: flashrt-fp8-swiglu-ffn

## BF16 Region Entry: RTX 5090 Source RC (2026-07-17)

- Torch: `2.9.0a0+145a3a7bda.nv25.10`
- Warmup/iterations/rounds: `20/100/5`; primary FlashRT-vs-BF16 timing uses
  A-B-B-A ordering and reports the median samples.
- All rows pass exact input quantization, staged numerical gates,
  `torch.compile(fullgraph=True)` for the FlashRT op, and explicit CUDA Graph
  replay. M=51 rows also pass the `>=1.3x` BF16 eager promotion gate.

### SwiGLU

| Shape | FlashRT us | Graph us | Separate quant us | Kernel-only us | BF16 eager us | vs eager | BF16 compile us | vs separate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decoder M8 | 16.436 | 16.395 | 32.945 | 14.377 | 27.576 | 1.68x | 37.485 | 2.00x |
| decoder M51 | 20.543 | 18.446 | 32.996 | 18.478 | 30.153 | 1.47x | 38.967 | 1.61x |
| decoder M64 | 20.547 | 18.439 | 32.953 | 18.510 | 31.963 | 1.56x | 39.387 | 1.60x |
| decoder M105 | 26.678 | 22.540 | 39.077 | 24.643 | 42.561 | 1.60x | 39.619 | 1.46x |
| decoder M128 | 26.674 | 22.533 | 39.066 | 24.621 | 42.813 | 1.61x | 39.159 | 1.46x |
| DiT M51 | 30.775 | 26.631 | 41.117 | 26.675 | 49.236 | 1.60x | 93.361 | 1.34x |
| DiT M128 | 45.070 | 36.882 | 55.441 | 40.991 | 71.642 | 1.59x | 115.129 | 1.23x |

### GeGLU

| Shape | FlashRT us | Graph us | Separate quant us | Kernel-only us | BF16 eager us | vs eager | BF16 compile us | vs separate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decoder M8 | 16.427 | 16.383 | 33.393 | 14.393 | 28.260 | 1.72x | 37.401 | 2.03x |
| decoder M51 | 20.543 | 18.440 | 32.992 | 18.487 | 29.018 | 1.41x | 38.984 | 1.61x |
| decoder M64 | 20.546 | 18.441 | 32.935 | 18.486 | 30.241 | 1.47x | 40.045 | 1.60x |
| decoder M105 | 26.678 | 22.534 | 39.070 | 24.641 | 42.149 | 1.58x | 39.855 | 1.46x |
| decoder M128 | 26.675 | 22.538 | 39.082 | 24.621 | 42.021 | 1.58x | 40.084 | 1.47x |
| DiT M51 | 30.734 | 26.635 | 41.126 | 26.592 | 47.964 | 1.56x | 92.519 | 1.34x |
| DiT M128 | 45.062 | 36.878 | 55.219 | 40.990 | 71.664 | 1.59x | 115.051 | 1.23x |

The BF16 baseline does not time FP8 dequantization. Random per-tensor FP8
weight quantization gives BF16-reference cosine `0.99783-0.99790`; migration
parity is checked separately against the established FP8 staged path.
These are source-extension release-candidate results, not Hub built-artifact
claims.

This is a second-batch package under active validation. It implements true
SwiGLU, not the GELU path from `flashrt-fp8-ffn`.

## Current Local Status

- Device: NVIDIA GeForce RTX 5090
- Backend: local source extension
- Commands:

  ```bash
  python flashrt-fp8-swiglu-ffn/tests/test_fp8_swiglu_ffn.py --backend source --mode smoke
  python flashrt-fp8-swiglu-ffn/tests/test_fp8_swiglu_ffn.py --backend source --mode full
  python flashrt-fp8-swiglu-ffn/benchmarks/benchmark.py --backend source --shapes all --warmup 3 --iters 10
  python flashrt-fp8-swiglu-ffn/benchmarks/benchmark.py --backend source --shapes headline --compile-baseline --warmup 3 --iters 10
  ```

- Correctness result: PASS
- Strict gate: fused `fp8_swiglu_mlp_bf16` output must match staged FlashRT
  primitives with `p99_abs=0`.
- PyTorch reference metrics are reported separately because FP8 GEMM reduction
  and FP8 requant boundaries can differ from cuBLASLt/FlashRT staged execution.

## Source-Extension Full Shape Sweep

| Shape | M,K,H,N | FlashRT us | Eager us | vs eager | Staged p99 | Staged cosine | Torch-ref p99 | Torch-ref cosine | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pi05_decoder_m1 | 1,1024,4096,1024 | 13.942 | 105.094 | 7.54x | 0.000000 | 1.00000000 | 0.000000 | 1.00000000 | PASS |
| pi05_decoder_m8 | 8,1024,4096,1024 | 14.762 | 135.757 | 9.20x | 0.000000 | 1.00000000 | 0.000000 | 1.00000000 | PASS |
| pi05_decoder_m10 | 10,1024,4096,1024 | 14.957 | 139.882 | 9.35x | 0.000000 | 1.00000000 | 0.000000 | 1.00000000 | PASS |
| pi05_decoder_m16 | 16,1024,4096,1024 | 16.640 | 154.157 | 9.26x | 0.000000 | 1.00000012 | 0.000000 | 1.00000000 | PASS |
| pi05_vision_1view | 256,1152,4304,1152 | 39.622 | 319.632 | 8.07x | 0.000000 | 1.00000012 | 0.000000 | 1.00000000 | PASS |
| pi05_vision_2view | 512,1152,4304,1152 | 68.304 | 460.109 | 6.74x | 0.000000 | 0.99999988 | 0.125000 | 1.00000000 | PASS |
| pi05_vision_3view | 768,1152,4304,1152 | 82.621 | 603.498 | 7.30x | 0.000000 | 0.99999994 | 0.000000 | 0.99999988 | PASS |
| groot_vl_seq512 | 512,2048,8192,2048 | 144.227 | 1364.010 | 9.46x | 0.000000 | 1.00000000 | 8.000000 | 0.99999988 | PASS |
| groot_vl_seq1024 | 1024,2048,8192,2048 | 283.830 | 2334.035 | 8.22x | 0.000000 | 1.00000000 | 8.000000 | 0.99999988 | PASS |
| groot_vl_seq2520 | 2520,2048,8192,2048 | 638.758 | 5319.674 | 8.33x | 0.000000 | 1.00000000 | 6.000000 | 0.99999994 | PASS |
| action_dit | 41,1536,6144,1536 | 25.290 | 337.571 | 13.35x | 0.000000 | 0.99999988 | 0.000000 | 1.00000000 | PASS |

## Source-Extension `torch.compile` Headline

The compile baseline is segmented compile-stable: FP8 dequant GEMM regions are
compiled, while the FP8 requant boundary remains eager to avoid invalid
comparison artifacts.

| Shape | M,K,H,N | FlashRT us | Eager us | vs eager | Compile us | vs compile | Compile status | Staged p99 | Torch-ref p99 | Status |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| pi05_decoder_m10 | 10,1024,4096,1024 | 15.075 | 135.853 | 9.01x | 144.979 | 9.62x | segmented-ok | 0.000000 | 0.000000 | PASS |
| pi05_vision_2view | 512,1152,4304,1152 | 68.173 | 466.707 | 6.85x | 431.334 | 6.33x | segmented-ok | 0.000000 | 3.000000 | PASS |
| groot_vl_seq1024 | 1024,2048,8192,2048 | 284.080 | 2343.046 | 8.25x | 2132.746 | 7.51x | segmented-ok | 0.000000 | 4.000000 | PASS |

## Required Before Publishing

The following rows are intentionally pending until they are generated by this
package's benchmark scripts:

| Stage | Command | Status |
|---|---|---|
| Source full correctness | `tests/test_fp8_swiglu_ffn.py --backend source --mode full` | done |
| Source benchmark headline | `benchmarks/benchmark.py --backend source --shapes headline` | done |
| Source benchmark full grid | `benchmarks/benchmark.py --backend source --shapes all` | done |
| Kernel-builder artifact build | `kernel-builder build-and-copy flashrt-fp8-swiglu-ffn` | pending |
| Built-artifact correctness | `tests/test_fp8_swiglu_ffn.py --backend installed --mode full` | pending |
| Built-artifact benchmark | `benchmarks/benchmark.py --backend installed --shapes all` | pending |
| Multi-hardware matrix | RTX 5090 plus external hardware | pending |

## Shape Grid

Initial VLA/PI0.5-oriented grid:

| Family | Shapes |
|---|---|
| PI0.5 decoder | M 1, 8, 10, 16; K 1024; H 4096; N 1024 |
| PI0.5 vision | M 256, 512, 768; K 1152; H 4304; N 1152 |
| GROOT/VL FFN | M 512, 1024, 2520; K 2048; H 8192; N 2048 |
| Action/DiT-shaped FFN | M 41; K 1536; H 6144; N 1536 |

Every published row should report runtime plus correctness metrics:
`max_abs`, `mean_abs`, `p99_abs`, cosine similarity, and p99 relative error
with `abs(reference)` floored at 1.0.
