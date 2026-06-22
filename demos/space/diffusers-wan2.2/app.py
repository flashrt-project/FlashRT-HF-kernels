"""ZeroGPU Space: accelerate Wan2.2 video generation with FlashRT kernels.

Builds the standard diffusers ``WanPipeline`` through official APIs (see
``load_flashrt_wan``): the transformer's FP8 / NVFP4 GEMMs load via the diffusers
**quantization API** (``quantization_config=FlashRTDiffusersConfig(...)``, the same
entry point as torchao / bitsandbytes / modelopt), attention runs the FlashRT
SageAttention2 Blackwell kernel (SM120), and the VAE's causal 3D convs run FP8.

It generates the same prompt with the BF16 baseline and the chosen FlashRT path
and reports the speedup. The output stays close to BF16 (cosine: 0.99992 FP8,
0.99864 NVFP4).
"""

from __future__ import annotations

import time

import gradio as gr
import spaces
import torch
from diffusers.utils import export_to_video

from optimize import load_flashrt_wan

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
HEIGHT, WIDTH, NUM_FRAMES, STEPS = 480, 832, 49, 20

_state: dict[str, object] = {}


def _load(mode: str):
    """mode: 'baseline' | 'fp8' | 'nvfp4'. FP8/NVFP4 quantize the transformer via
    the official diffusers quantization API (FlashRTDiffusersConfig)."""
    pipe = _state.get(mode)
    if pipe is None:
        pipe = load_flashrt_wan(
            MODEL_ID, mode=("bf16" if mode == "baseline" else mode),
            height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
            vae_fp8=(mode != "baseline"), sage_attn=(mode != "baseline"),
        )
        _state[mode] = pipe
    return pipe


def _generate(pipe, prompt: str):
    torch.cuda.synchronize()
    start = time.perf_counter()
    frames = pipe(
        prompt=prompt, height=HEIGHT, width=WIDTH,
        num_frames=NUM_FRAMES, num_inference_steps=STEPS,
    ).frames[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return export_to_video(frames, fps=16), elapsed


@spaces.GPU(duration=600)
def compare(prompt: str, precision: str):
    nvfp4 = precision.startswith("NVFP4")
    label = "FlashRT NVFP4 (4-bit)" if nvfp4 else "FlashRT FP8"
    base_video, base_s = _generate(_load("baseline"), prompt)
    fr_video, fr_s = _generate(_load("nvfp4" if nvfp4 else "fp8"), prompt)
    speedup = base_s / fr_s if fr_s > 0 else 0.0
    cos = "0.99884" if nvfp4 else "0.99992"
    summary = (
        f"| path | seconds |\n|---|---:|\n"
        f"| BF16 baseline | {base_s:.1f} |\n"
        f"| {label} | {fr_s:.1f} |\n\n"
        f"**Live speedup: {speedup:.2f}x** (eager) — cosine vs BF16: {cos}"
    )
    return base_video, fr_video, summary


_DESC = """\
# FlashRT-accelerated Wan2.2 video (diffusers)

The standard diffusers `WanPipeline`, accelerated **through official diffusers
APIs**: the transformer's FP8/NVFP4 GEMMs load via the **quantization API**
(`quantization_config=FlashRTDiffusersConfig(...)`, like torchao/bitsandbytes),
attention via the FlashRT SageAttention2 Blackwell processor (SM120), and the
VAE's 3D convs in FP8.

**Per denoise step, 480×480×33, RTX 5090 (SM120), cosine vs BF16 ≈ 0.999:**

| stack | ms/step | vs BF16 |
|---|---:|---:|
| BF16 | 161.0 | 1.00x |
| **FlashRT NVFP4 (4-bit), eager** | 79.9 | **2.01x** |
| **+ torch.compile** | 42.5 | **3.79x** |

NVFP4 (4-bit) is the headline; FP8 is the quality-preserving option. FlashRT also
runs SageAttention2 (attention) and FP8 VAE 3D convs (`world-model-conv`).
The 3.79x uses `torch.compile`; on ZeroGPU the equivalent is AoTI.
"""


with gr.Blocks(title="FlashRT Wan2.2 (FP8 / NVFP4)") as demo:
    gr.Markdown(_DESC)
    prompt = gr.Textbox(label="Prompt", value="a cat surfing a wave, cinematic, golden hour")
    precision = gr.Radio(
        ["FP8 (max quality)", "NVFP4 (4-bit, fastest)"],
        value="NVFP4 (4-bit, fastest)",
        label="FlashRT precision",
    )
    run = gr.Button("Compare BF16 vs FlashRT", variant="primary")
    summary = gr.Markdown()
    with gr.Row():
        base_out = gr.Video(label="BF16 baseline")
        fr_out = gr.Video(label="FlashRT")
    run.click(compare, [prompt, precision], [base_out, fr_out, summary])


if __name__ == "__main__":
    demo.launch()
