---
title: FlashRT Fast Wan2.2
emoji: 🎬
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# FlashRT Fast Wan2.2 (diffusers)

Wan2.2's denoiser is a diffusion transformer (DiT). FlashRT accelerates it
**through official diffusers APIs** — no hand-patching:

| step | official API | provider |
|---|---|---|
| FP8 / NVFP4 transformer GEMMs | **quantization API** (`quantization_config`) | FlashRT (`flashrt/fp4-gemm`, `flashrt/flashrt-fp8-ffn`) |
| attention | `set_attn_processor` (attention-processor API) | FlashRT (`flashrt/sageattention2-blackwell`, SM120) |
| VAE causal 3D convs (FP8) | module swap (Conv3d, not Linear) | FlashRT (`flashrt/world-model-conv`) |
| `channels_last` | diffusers-native | — |

The FP8 / NVFP4 GEMMs load via the diffusers **quantization API**, the same entry
point as torchao / bitsandbytes / modelopt — FlashRT is registered as a
quantization backend whose linear forward calls the Hub kernels:

```python
from diffusers import WanTransformer3DModel, WanPipeline
from flashrt_diffusers_quantizer import FlashRTDiffusersConfig   # registers the backend

transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="transformer", torch_dtype="bfloat16",
    quantization_config=FlashRTDiffusersConfig(mode="nvfp4"))   # FP8/NVFP4 via the quant API
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers", transformer=transformer, torch_dtype="bfloat16").to("cuda")
# or simply: from optimize import load_flashrt_wan;  pipe = load_flashrt_wan(model_id, mode="nvfp4")
```

## Measured (RTX 5090, per denoise step, 480², eager)

| stack | speedup | cosine vs BF16 |
|---|---:|---:|
| BF16 | 1.00x | — |
| **FlashRT FP8** | **1.28x** | 0.99989 |
| **FlashRT NVFP4** (4-bit) | **1.97x** | 0.99864 |

- **NVFP4** (the headline) runs the transformer GEMMs in 4-bit (`flashrt/fp4-gemm`),
  the same dynamic-per-block NVFP4 path FlashRT's production runtime uses.
- **FP8** uses a generic per-Linear quantizer (a Linear-level quantizer can't fuse
  the GELU MLP, so it trails a hand-fused path — the cost of going through the
  official API; NVFP4, the headline, is unaffected).
- **VAE** convs run FP8 (`flashrt/world-model-conv`, decode cosine 0.9998) — Conv3d,
  so swapped directly (not covered by the Linear quantizer).
- **Attention** runs the FlashRT SageAttention2 Blackwell kernel
  (`flashrt/sageattention2-blackwell`, INT8-QK / FP8-PV prefill, head_dim 128) via
  the `set_attn_processor` API — quantized self/cross-attention on SM120.

The Space runs **eager** (ZeroGPU does not support `torch.compile`).

## Hardware

Targets NVIDIA Blackwell (SM120); runs on the ZeroGPU RTX PRO 6000 pool.
