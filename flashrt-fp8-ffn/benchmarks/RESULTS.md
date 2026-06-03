# Benchmark Results: flashrt-fp8-ffn

## RTX 5090 Source-Extension Results

- Device: NVIDIA GeForce RTX 5090
- Compute capability: SM120
- Torch: 2.9.1+cu128
- Backend: local source extension
- Baselines: PyTorch eager and, for headline rows,
  `torch.compile(fullgraph=True, mode="reduce-overhead")`
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

These rows compare the full FP8 GELU MLP sublayer against both PyTorch eager
and `torch.compile`.

| Shape | M,K,H,N | Layers | FlashRT us | Eager us | vs eager | Compile us | vs compile | P99 abs | P99 rel | Max abs | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pi05_decoder_ffn_m10 | 10,1024,4096,1024 | 18 | 368.592 | 2435.814 | 6.61x | 1410.750 | 3.83x | 0.0000 | 0.000000 | 0.0625 | PASS |
| pi05_vision_ffn_2view | 512,1152,4304,1152 | 27 | 1679.115 | 10778.622 | 6.42x | 8318.347 | 4.95x | 0.0000 | 0.000000 | 32.0000 | PASS |
| groot_vit_ffn_2view | 512,1024,4096,1024 | 24 | 1176.670 | 8271.027 | 7.03x | 6409.768 | 5.45x | 0.0000 | 0.000000 | 16.0000 | PASS |
| groot_vl_self_attn_ffn_seq1024 | 1024,2048,8192,2048 | 4 | 954.864 | 6359.513 | 6.66x | 5369.453 | 5.62x | 0.0000 | 0.000000 | 64.0000 | PASS |

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

Built-artifact and multi-hardware results are still pending.
