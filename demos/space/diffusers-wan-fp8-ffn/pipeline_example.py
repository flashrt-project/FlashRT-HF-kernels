"""Minimal Wan2.2 pipeline example — FlashRT via the official diffusers quant API.

FlashRT registers as a diffusers quantization backend, so the transformer's
FP8 / NVFP4 GEMMs load through `quantization_config` (the same entry point as
torchao / bitsandbytes), and `torch.compile` then gives the largest per-step win.

    python pipeline_example.py [bf16|fp8|nvfp4]   # default nvfp4

Measured per denoise step, 480², RTX 5090 (SM120):
    bf16  161 ms (1.00x)
    fp8    67 ms (2.40x, torch.compile + static-scale; quality-preserving, cos 0.9999)
    nvfp4  44 ms (3.63x, torch.compile) / 81 ms (1.97x, eager)
"""

import sys
import time

import torch
from diffusers.utils import export_to_video

from optimize import load_flashrt_wan

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
H, W, FRAMES, STEPS = 480, 480, 33, 20


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "nvfp4"
    print(f"loading Wan2.2 via the official diffusers quantization API (mode={mode}) ...")

    # One call: quantized transformer (quantization_config) + FA2 attention + FP8 VAE
    # + torch.compile (auto-on for nvfp4). See optimize.load_flashrt_wan.
    pipe = load_flashrt_wan(MODEL_ID, mode=mode, height=H, width=W, num_frames=FRAMES)

    prompt = "a cat surfing a wave, cinematic, golden hour"
    # warmup (also triggers torch.compile / VAE calibration)
    pipe(prompt=prompt, height=H, width=W, num_frames=FRAMES, num_inference_steps=2)

    torch.cuda.synchronize()
    t = time.perf_counter()
    frames = pipe(prompt=prompt, height=H, width=W, num_frames=FRAMES,
                  num_inference_steps=STEPS).frames[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t

    path = export_to_video(frames, fps=16)
    print(f"[{mode}] {len(frames)} frames, {STEPS} steps in {dt:.1f}s "
          f"({dt / STEPS * 1e3:.0f} ms/step) -> {path}")


if __name__ == "__main__":
    main()
