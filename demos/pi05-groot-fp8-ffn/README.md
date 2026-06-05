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
python flashrt-fp8-ffn/examples/fp8_gelu_mlp_block.py --compile
python scripts/torch_compile_smoke.py --version 1
python flashrt-fp8-ffn/tests/test_fp8_ffn.py --backend source
python flashrt-fp8-ffn/benchmarks/benchmark.py \
  --backend source \
  --shapes headline \
  --compile-baseline \
  --warmup 5 \
  --iters 20 \
  --output internal-tests/demos/pi05-groot-fp8-ffn/results-rtx5090.json \
  --markdown internal-tests/demos/pi05-groot-fp8-ffn/results-rtx5090.md
```

To benchmark the published Hugging Face Kernel Hub package directly:

```bash
python flashrt-fp8-ffn/benchmarks/benchmark.py \
  --backend hub \
  --shapes headline \
  --compile-baseline \
  --warmup 3 \
  --iters 10 \
  --output internal-tests/demos/pi05-groot-fp8-ffn-hub/results-rtx5090-torch211.json \
  --markdown internal-tests/demos/pi05-groot-fp8-ffn-hub/results-rtx5090-torch211.md
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
| PI0.5 decoder FFN | `10,1024,4096,1024` | 18 | 6.61x | 3.83x | PASS |
| PI0.5 vision FFN | `512,1152,4304,1152` | 27 | 6.42x | 4.95x | PASS |
| GROOT ViT FFN | `512,1024,4096,1024` | 24 | 7.03x | 5.45x | PASS |
| GROOT VL self-attn FFN | `1024,2048,8192,2048` | 4 | 6.66x | 5.62x | PASS |

Expanded source-extension coverage also passes:

- PI0.5 decoder chunks: `M in {1,8,10,16}`.
- PI0.5 vision: 1/2/3 views.
- GROOT ViT: 1/2/4 views.
- GROOT DeepStack merger.
- GROOT VL self-attn FFN: `seq in {512,1024,2520}`.
- GROOT action DiT GELU FFN shape.

## Interpretation

This is a full FFN model-block benchmark, not a full PI0.5/GROOT generation
benchmark. It proves that a complete, Hub-loadable FlashRT FP8 FFN block keeps
a multi-x gap over both PyTorch eager and `torch.compile` on VLA/VLM-shaped
workloads. Full checkpoint-level throughput should be reported separately once
the model frontend, calibration, and input pipeline are fixed for the run.

The full serving stack can stack this FFN acceleration with attention kernels,
QKV postprocess, quant/layout kernels, CUDA Graph orchestration, and community
techniques such as distillation or fewer denoising steps.

## RTX 5090 Hub Results

Published package:
`get_kernel("flashrt/flashrt-fp8-ffn", version=1, trust_remote_code=True)`.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- Torch: `2.11.0+cu128`
- Benchmark mode: Hub kernel, `warmup=3`, `iters=10`
- Baselines: PyTorch eager FP8 reference and compile-stable
  `torch.compile` reference

| Block | Shape | Layers | FlashRT Hub us | Eager us | vs eager | Compile us | vs compile | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PI0.5 decoder FFN | `10,1024,4096,1024` | 18 | 367.997 | 2435.322 | 6.62x | 2301.635 | 6.25x | PASS |
| PI0.5 vision FFN | `512,1152,4304,1152` | 27 | 1677.056 | 10783.178 | 6.43x | 10146.278 | 6.05x | PASS |
| GROOT ViT FFN | `512,1024,4096,1024` | 24 | 1179.792 | 8492.259 | 7.20x | 7875.162 | 6.68x | PASS |
| GROOT VL self-attn FFN | `1024,2048,8192,2048` | 4 | 982.938 | 6444.928 | 6.56x | 6010.432 | 6.11x | PASS |

This is a complete FFN sublayer stack benchmark for PI0.5/GROOT-shaped
workloads, not a full checkpoint generation benchmark. Full model E2E should
use the upstream FlashRT runtime once model loading, calibration, tokenizer or
vision preprocessing, and scheduling are fixed for the target machine.
