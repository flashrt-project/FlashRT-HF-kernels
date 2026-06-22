---
title: FlashRT FP8 / NVFP4 Transformer Blocks
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# FlashRT on transformers (Qwen3-8B) — through the official APIs

FlashRT plugs into a standard `transformers` causal LM **only through official
APIs** — no hand-patching of the model. Each step is a named FlashRT Hub kernel
reached through an official mechanism:

| step | official transformers API | FlashRT kernel |
|---|---|---|
| NVFP4 / FP8 GEMMs | quantization API (`quantization_config`) | `flashrt/fp4-gemm` (NVFP4) · `flashrt/flashrt-fp8-ffn` (FP8) |
| RMSNorm | `kernelize()` (the gpt-oss path) | `flashrt/flashrt-residual-norm-quant` |
| prefill attention | `attn_implementation="sage2_blackwell"` (`AttentionInterface.register`) | `flashrt/sageattention2-blackwell` |

## How to run

```bash
pip install -r requirements.txt
python serving_example.py nvfp4         # or: bf16 | fp8   (default nvfp4)
```

`serving_example.py` loads Qwen3-8B through the official APIs and prints prefill
latency + greedy-decode tok/s. The same stack wrapped in a Gradio / ZeroGPU UI is
in `app.py`. From code:

```python
from transformers import AutoModelForCausalLM, kernelize
from flashrt_quantizer import FlashRTConfig          # registers the quantization backend
import flashrt_sage_attn                              # registers attn_implementation="sage2_blackwell"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=FlashRTConfig(mode="nvfp4"),  # NVFP4 GEMMs via the official quant API
    attn_implementation="sage2_blackwell",            # SageAttention2 prefill (decode -> SDPA)
    dtype="bfloat16", device_map="cuda",
)
kernelize(model)                                      # RMSNorm -> FlashRT via the official path
```

## Measured (RTX 5090, Qwen3-8B)

Prefill (1024 tokens):

| stack | prefill | vs BF16 |
|---|---:|---:|
| BF16 | 98.3 ms | 1.00x |
| + NVFP4 GEMMs | 37.9 ms | **2.59x** |
| + kernelize RMSNorm | 33.5 ms | **2.93x** |

- **Decode** (greedy, tok/s): BF16 66.4 → full FlashRT stack **86.7 (1.31x)** — the
  gain is from NVFP4 weights + `kernelize` RMSNorm.
- **SageAttention2** is a prefill kernel; at 1024 tokens attention is negligible,
  so its win shows at longer context (prefill 4096 reaches **3.32x** with
  `torch.compile`: BF16 379 → 114 ms). Decode (q_len=1) falls back to SDPA.
- cosine ≈ 0.999 (NVFP4 is the same dynamic-per-block 4-bit path FlashRT's
  production runtime uses); FP8 (`mode="fp8"`) is the quality-preserving option.

`kernelize` uses the same `register_kernel_mapping` + `kernelize` mechanism as
gpt-oss (RMSNorm → Liger, MoE → MegaBlocks), here pointed at FlashRT kernel-layers.
Because RMSNorm is a decorated layer in ~every modern decoder, one
`kernelize(model)` call covers the whole model and transfers across the model zoo
(Llama / Qwen / Mistral / GLM / DeepSeek / gpt-oss …).

## Scope (read this)

Honest about where the wins are:

- Quantized weights cut memory traffic and (4-bit) compute, so the largest wins are
  in the **compute-bound regime** — prefill (2.9x at 1024 tok, 3.3x at 4096 +
  compile).
- **Decode** improves a real 1.31x here (NVFP4 + kernelize), but peak single-stream
  decode is launch- and bandwidth-bound; reaching it needs a fully-fused runtime
  with persistent buffers, CUDA-graph capture, and a dedicated small-M decode GEMM
  — i.e. the FlashRT serving runtime, not this demo.
- Activations are quantized dynamically (no calibration); the GEMMs are
  math-equivalent to the BF16 reference up to FP8 / NVFP4 rounding.

The companion `diffusers` demo (Wan2.2 video) is in
[`../diffusers-wan2.2`](../diffusers-wan2.2), where the same kernel
family accelerates the DiT, attention, and VAE.

## Hardware

The NVFP4 (4-bit) path needs **NVIDIA Blackwell (SM120)** tensor cores — RTX 5090
or the ZeroGPU RTX PRO 6000. Pre-Blackwell GPUs cannot run the FP4 path. Requires
`kernels` 0.12.x and a Blackwell `torch211-cu128` build of each kernel.
