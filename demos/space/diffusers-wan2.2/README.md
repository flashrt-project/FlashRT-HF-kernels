---
title: FlashRT Fast Wan2.2
emoji: 🎬
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# FlashRT-accelerated Wan2.2 (diffusers)

Wan2.2 TI2V-5B's denoiser is a diffusion transformer (DiT). FlashRT accelerates it
**through official `diffusers` APIs** — no forked libraries, no hand-patching of
the model. Each step is a named FlashRT Hub kernel reached through an official
mechanism:

| step | official diffusers API | FlashRT kernel |
|---|---|---|
| NVFP4 / FP8 transformer GEMMs | quantization API (`quantization_config`) | `flashrt/fp4-gemm` (NVFP4) · `flashrt/flashrt-fp8-ffn` (FP8) |
| self / cross attention | `set_attn_processor` | `flashrt/sageattention2-blackwell` (INT8-QK / FP8-PV prefill, head_dim 128) |
| VAE causal 3D convs | module swap (Conv3d) | `flashrt/world-model-conv` (FP8) |

## How to run

```bash
pip install -r requirements.txt
python pipeline_example.py nvfp4        # or: bf16 | fp8   (default nvfp4)
```

`pipeline_example.py` builds the pipeline through the official APIs and prints the
BF16-vs-FlashRT per-step time. The same stack wrapped in a Gradio / ZeroGPU UI is
in `app.py`. To use it from code:

```python
from optimize import load_flashrt_wan
pipe = load_flashrt_wan("Wan-AI/Wan2.2-TI2V-5B-Diffusers", mode="nvfp4")
```

or through the raw diffusers quantization API:

```python
from diffusers import WanTransformer3DModel, WanPipeline
from flashrt_diffusers_quantizer import FlashRTDiffusersConfig   # registers the backend
from flashrt_wan_attn import WanSageAttention2Processor

transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="transformer", torch_dtype="bfloat16",
    quantization_config=FlashRTDiffusersConfig(mode="nvfp4"))   # NVFP4 GEMMs via the quant API
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers", transformer=transformer, torch_dtype="bfloat16").to("cuda")
pipe.transformer.set_attn_processor(WanSageAttention2Processor())   # SageAttention2
```

## Measured (RTX 5090, per denoise step, 480×480×33)

| step | ms/step | vs BF16 |
|---|---:|---:|
| BF16 | 161.0 | 1.00x |
| + SageAttention2 (`set_attn_processor`) | 158.7 | 1.01x |
| + NVFP4 GEMMs (`quantization_config`) | 79.9 | **2.01x** |
| + `torch.compile` | 42.5 | **3.79x** |

cosine vs BF16 ≈ 0.999.

- **NVFP4 (4-bit) is the headline** — the same dynamic-per-block path FlashRT's
  production runtime uses. The eager NVFP4 step is **2.01x** (what ZeroGPU runs);
  the **3.79x** adds `torch.compile`, a local-GPU number (Spaces run eager).
- **FP8** is available as the quality-preserving precision (`mode="fp8"`,
  cosine ≈ 0.9999). A Linear-level quantizer can't fuse the GELU MLP, so FP8
  trails a hand-fused path — the cost of going through the official API; NVFP4 is
  unaffected.
- **SageAttention2** is a prefill kernel (~1.25x on Wan's self-attention at this
  shape); at the per-step level attention is a small fraction, so its E2E step
  contribution is small (-2.3 ms).
- **VAE** 3D convs run FP8 via `world-model-conv` on the decode side (not in the
  per-step number above).

## Hardware

The NVFP4 (4-bit) path needs **NVIDIA Blackwell (SM120)** tensor cores — RTX 5090
or the ZeroGPU RTX PRO 6000. Pre-Blackwell GPUs cannot run the FP4 path. Requires
`kernels` 0.12.x and a Blackwell `torch211-cu128` build of each kernel.
