"""Build a FlashRT-accelerated Wan2.2 diffusers pipeline through official APIs.

FlashRT plugs in as a **diffusers quantization backend**: the transformer's
FP8 / NVFP4 GEMMs load via
``WanTransformer3DModel.from_pretrained(quantization_config=FlashRTDiffusersConfig(...))``
— the same entry point as torchao / bitsandbytes / modelopt. Attention runs the
FlashRT SageAttention2 Blackwell kernel (INT8-QK / FP8-PV prefill, head_dim 128)
via the diffusers attention-processor API, and the VAE's causal 3D convs (Conv3d,
not Linear) optionally run FP8 via the world-model-conv kernel.
"""

from __future__ import annotations

import torch
from diffusers import WanPipeline, WanTransformer3DModel


def load_flashrt_wan(
    model_id: str,
    *,
    mode: str = "nvfp4",          # "nvfp4" | "fp8" | "bf16"
    height: int = 480,
    width: int = 832,
    num_frames: int = 49,
    vae_fp8: bool = True,
    sage_attn: bool = True,
    channels_last: bool = True,
    compile_transformer: bool | None = None,
):
    """Load a Wan2.2 pipeline with FlashRT applied through official diffusers APIs.

    The transformer GEMMs are quantized at load time via the diffusers
    quantization API (``quantization_config``); attention uses the FlashRT
    SageAttention2 Blackwell kernel (INT8-QK / FP8-PV, head_dim 128) via the
    diffusers attention-processor API; the VAE's 3D convs optionally run FP8.

    ``compile_transformer`` torch.compiles the transformer for the largest per-step
    win. Default (``None``) auto-enables it for NVFP4 (where it gives ~3.6x) and
    leaves it off for FP8 (whose dynamic-scale forward does not benefit under
    compile).
    """

    if compile_transformer is None:
        compile_transformer = mode in ("nvfp4", "fp8")

    transformer = None
    if mode in ("fp8", "nvfp4"):
        from flashrt_diffusers_quantizer import FlashRTDiffusersConfig
        transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
            quantization_config=FlashRTDiffusersConfig(mode=mode),
        )

    extra = {"transformer": transformer} if transformer is not None else {}
    pipe = WanPipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, **extra).to("cuda")

    if sage_attn:
        from flashrt_wan_attn import WanSageAttention2Processor
        pipe.transformer.set_attn_processor(WanSageAttention2Processor())

    if channels_last and hasattr(pipe, "vae"):
        try:  # 3D-conv VAE (5D tensors) — channels_last (4D) is a no-op there
            pipe.vae.to(memory_format=torch.channels_last)
        except RuntimeError:
            pass

    if vae_fp8 and hasattr(pipe, "vae"):
        # FP8 Wan VAE causal-3D convs (Conv3d — not covered by the Linear quantizer)
        from flashrt_vae_conv import collect_vae_conv_inputs, patch_wan_vae_conv

        def _calib():
            pipe(prompt="a calibration clip of gentle waves", height=height,
                 width=width, num_frames=num_frames, num_inference_steps=1)

        vae_inputs = collect_vae_conv_inputs(pipe.vae, _calib)
        patch_wan_vae_conv(pipe.vae, vae_inputs)

    if mode == "fp8":
        # Static FP8 activation scales (compile-friendly; no per-step amax)
        from flashrt_diffusers_quantizer import calibrate_fp8
        calibrate_fp8(pipe.transformer, lambda: pipe(
            prompt="a calibration clip of gentle waves", height=height,
            width=width, num_frames=num_frames, num_inference_steps=1))

    if compile_transformer:
        import torch._dynamo as dyn
        dyn.config.cache_size_limit = 64
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune-no-cudagraphs", fullgraph=False)

    return pipe
