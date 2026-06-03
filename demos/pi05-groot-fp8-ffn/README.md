# PI0.5/GROOT FP8 FFN Demo

This demo is the first-version VLA/VLM showcase for `flashrt-fp8-ffn`.

It measures complete model-shaped FFN sublayers:

```text
FP8 up GEMM -> bias/GELU -> FP8 quant -> FP8 down GEMM -> bias
```

This is stronger than the epilogue-only demo because it includes both FP8 GEMMs
and the activation/quantization bridge between them.

## Run

```bash
python flashrt-fp8-ffn/tests/test_fp8_ffn.py --backend source
python flashrt-fp8-ffn/benchmarks/benchmark.py \
  --backend source \
  --shapes pi05_decoder_ffn,groot_vit_ffn_2view,groot_vl_self_attn_ffn \
  --compile-baseline \
  --warmup 5 \
  --iters 20 \
  --output internal-tests/demos/pi05-groot-fp8-ffn/results-rtx5090.json \
  --markdown internal-tests/demos/pi05-groot-fp8-ffn/results-rtx5090.md
```

For the very large VL self-attn shape, use fewer iterations if the machine is
busy:

```bash
python flashrt-fp8-ffn/benchmarks/benchmark.py \
  --backend source \
  --shapes groot_vl_self_attn_ffn \
  --compile-baseline \
  --warmup 3 \
  --iters 10
```

## RTX 5090 Source Results

| Block | Shape | Layers | vs eager | vs compile | Precision |
| --- | ---: | ---: | ---: | ---: | --- |
| PI0.5 decoder FFN | `10,1024,4096,1024` | 18 | 6.62x | 3.83x | PASS |
| GROOT ViT FFN | `512,1024,4096,1024` | 24 | 7.19x | 5.31x | PASS |
| GROOT VL self-attn FFN | `1024,2048,8192,2048` | 4 | 6.58x | 5.57x | PASS |

## Interpretation

This is a full FFN sublayer benchmark, not a full PI0.5/GROOT generation
benchmark. It proves that a complete, Hub-loadable FlashRT FP8 FFN block keeps
a multi-x gap over both PyTorch eager and `torch.compile` on VLA/VLM-shaped
workloads.

The full serving stack can stack this FFN acceleration with attention kernels,
QKV postprocess, quant/layout kernels, CUDA Graph orchestration, and community
techniques such as distillation or fewer denoising steps.
