---
title: FlashRT FP8 / NVFP4 Transformer Blocks
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# FlashRT FP8 / NVFP4 on transformers — through the official APIs

FlashRT plugs into a standard `transformers` causal LM **only through official
APIs** — no hand-patching of the model:

- **FP8 / NVFP4 GEMMs** load via the transformers **quantization API**
  (`from_pretrained(quantization_config=FlashRTConfig(...))`) — the same entry
  point as torchao / bitsandbytes / compressed-tensors. FlashRT is registered as
  a quantization backend whose linear forward calls the Hub kernels
  ([`flashrt/fp4-gemm`](https://huggingface.co/flashrt/fp4-gemm),
  [`flashrt/flashrt-fp8-ffn`](https://huggingface.co/flashrt/flashrt-fp8-ffn)).
- **RMSNorm** is swapped via the official `kernelize()` mechanism (the gpt-oss path).
- **Attention** uses the official `attn_implementation` Hub-kernel backend.

## Measured (RTX 5090, Qwen3-8B, prefill 1024 tokens)

| stack | prefill | vs BF16 |
|---|---:|---:|
| BF16 | 98.3 ms | 1.00x |
| **FlashRT FP8** (default) | 65.5 ms | **1.50x** |
| **FlashRT NVFP4** (4-bit) | 33.7 ms | **2.91x** |

Both go through `quantization_config` + `kernelize`. FP8 preserves perplexity;
NVFP4 is the same dynamic-per-block 4-bit path FlashRT's production runtime uses.
The Space reports live perplexity for both on the same passage.

```python
from transformers import AutoModelForCausalLM, kernelize
from flashrt_quantizer import FlashRTConfig          # registers the FlashRT quantizer
import flashrt_sage_attn                              # registers attn_implementation="sage2_blackwell"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=FlashRTConfig(mode="nvfp4"),  # FP8/NVFP4 GEMMs via the official quant API
    attn_implementation="sage2_blackwell",              # SageAttention2 prefill (import flashrt_sage_attn)
    dtype="bfloat16", device_map="cuda",
)
kernelize(model)                                      # RMSNorm -> FlashRT via the official path
```

The RMSNorm path uses the same `register_kernel_mapping` + `kernelize` mechanism
that gpt-oss uses (RMSNorm -> Liger, MoE -> MegaBlocks), here pointed at FlashRT
kernel-layers. Because RMSNorm is a decorated layer in ~every modern model, a
single `kernelize(model)` call covers the whole model and transfers across the
model zoo (Llama / Qwen / Mistral / Mixtral / GLM / DeepSeek / gpt-oss ...).

## Scope (read this)

This is an **integration + quality** demo, and it is honest about where FP8 helps:

- FP8/NVFP4 weights cut memory traffic and (4-bit) compute, so the win shows in
  the **compute-bound / large-batch regime** — prefill and throughput. On RTX 5090,
  prefill of 1024 tokens is **1.50x** faster with FP8 and **2.91x** with NVFP4.
- **Single-stream decode (M=1) is not the headline here.** At one token per step
  the path is launch- and bandwidth-bound, and `generate` on a quantized model
  cannot reach peak decode throughput. That regime needs a fully-fused runtime
  with persistent buffers, CUDA-graph capture, and a dedicated small-M decode GEMM
  — i.e. the FlashRT serving runtime, not this Space.
- Activations are quantized dynamically (no calibration); the GEMMs are
  math-equivalent to the BF16 reference up to FP8 / NVFP4 rounding.

## Companion demo (diffusers)

This Space shows FlashRT kernels dropping into `transformers` through the official
`kernels` library. The companion is the
[FlashRT Fast Wan2.2](https://huggingface.co/spaces/flashrt/flashrt-fast-wan22)
`diffusers` Space, where the same kernel family accelerates Wan2.2 video with FP8
and 4-bit NVFP4 GEMMs on Blackwell.

## Hardware

The kernels target NVIDIA Blackwell (SM120) and run on the ZeroGPU RTX PRO 6000
Blackwell pool.
