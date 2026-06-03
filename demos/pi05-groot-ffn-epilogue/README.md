# PI0.5/GROOT FFN FP8 Epilogue Demo

This demo benchmarks model-shaped FFN epilogue blocks from PI0.5 and GROOT
against exact PyTorch references:

- `bias + GELU(tanh) + static FP8 cast`
- `GELU(tanh) + static FP8 cast`
- `channel scale + static FP8 cast`

The benchmark is intentionally scoped to reusable model blocks. It is not a
full PI0.5 or GROOT generation benchmark, and it does not include the full FP8
GEMM or attention path. Those larger end-to-end demos should use the next
package once the FP8 GEMM/megakernel path is exported.

## Why These Shapes

| Block | Model surface | Shape | Repeats |
|---|---|---:|---:|
| `pi05_vision_ffn_2view` | PI0.5 SigLIP FFN fc1 epilogue | `512 x 4304` | 27 |
| `pi05_encoder_channel_scale` | PI0.5 encoder activation FP8 cast | `560 x 2048` | 18 |
| `groot_vit_ffn_2view` | GROOT ViT FFN fc1 epilogue | `512 x 4096` | 24 |
| `groot_deepstack_merge` | GROOT DeepStack merger fc1 epilogue | `128 x 4096` | 3 |
| `groot_vl_self_attn_ffn_long` | GROOT VL self-attn FFN fc1 epilogue | `1024 x 8192` | 4 |

## Run

```bash
python demos/pi05-groot-ffn-epilogue/benchmark.py \
  --backend source \
  --compile-baseline \
  --warmup 10 \
  --iters 50 \
  --output internal-tests/demos/pi05-groot-ffn-epilogue/results-rtx5090.json \
  --markdown internal-tests/demos/pi05-groot-ffn-epilogue/results-rtx5090.md
```

After publishing to the Hub, use:

```bash
python demos/pi05-groot-ffn-epilogue/benchmark.py \
  --backend hub \
  --repo-id flashrt/flashrt-gemm-epilogues \
  --version 1 \
  --compile-baseline
```

## Interpretation

Use this with the Wan QKV postprocess demo as the first-version story:

- Wan shows a VLA/video attention-preprocess pain point.
- This demo shows a repeated FFN epilogue/activation-quant pain point in
  PI0.5/GROOT-shaped backbones.

For a true end-to-end PI0.5/GROOT claim, package the full FP8 GEMM/FFN or
megakernel path and benchmark the complete model stage with the checkpoint and
calibration path enabled.
