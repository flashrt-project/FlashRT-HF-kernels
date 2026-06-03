# Benchmark Results: flashrt-fp8-ffn

## RTX 5090 Source-Extension Results

- Device: NVIDIA GeForce RTX 5090
- Torch: 2.9.1+cu128
- Backend: local source extension
- Baselines: PyTorch eager and `torch.compile(fullgraph=True, mode="reduce-overhead")`
- Precision gate: p99 absolute error <= 0.5 and p99 relative error with
  `abs(reference)` floored at 1.0 <= 0.05.

Commands:

```bash
python flashrt-fp8-ffn/tests/test_fp8_ffn.py --backend source
python flashrt-fp8-ffn/benchmarks/benchmark.py \
  --backend source \
  --shapes pi05_decoder_ffn,groot_vit_ffn_2view \
  --compile-baseline \
  --warmup 5 \
  --iters 20 \
  --output internal-tests/flashrt-fp8-ffn/benchmark-source-rtx5090.json \
  --markdown internal-tests/flashrt-fp8-ffn/benchmark-source-rtx5090.md
python flashrt-fp8-ffn/benchmarks/benchmark.py \
  --backend source \
  --shapes groot_vl_self_attn_ffn \
  --compile-baseline \
  --warmup 3 \
  --iters 10 \
  --output internal-tests/flashrt-fp8-ffn/benchmark-source-rtx5090-vl.json \
  --markdown internal-tests/flashrt-fp8-ffn/benchmark-source-rtx5090-vl.md
```

| Shape | M,K,H,N | Layers | FlashRT us | Eager us | vs eager | Compile us | vs compile | P99 abs | Max abs | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pi05_decoder_ffn` | 10,1024,4096,1024 | 18 | 368.224 | 2436.358 | 6.62x | 1409.782 | 3.83x | 0.0000 | 0.0625 | PASS |
| `groot_vit_ffn_2view` | 512,1024,4096,1024 | 24 | 1179.576 | 8477.020 | 7.19x | 6264.816 | 5.31x | 0.0000 | 32.0000 | PASS |
| `groot_vl_self_attn_ffn` | 1024,2048,8192,2048 | 4 | 977.558 | 6436.419 | 6.58x | 5447.811 | 5.57x | 0.0000 | 64.0000 | PASS |

## Interpretation

This package measures complete FP8 GELU MLP sublayers, not epilogue-only
fragments. It is a stronger first-version showcase surface for VLA/VLM
backbones because `torch.compile` still leaves a multi-x gap on repeated
PI0.5/GROOT-shaped FFN blocks.

Built-artifact and multi-hardware results are still pending.
