# flashrt-vla-video

Draft package for reusable VLA, vision, video, and diffusion kernels from
FlashRT.

This package should focus on gaps that are not already covered by common LLM
attention, MoE, and quantization packages.

This is the strongest first showcase candidate after `flashrt-gemm-epilogues`
if we can attach each API to a clear HF-style model-block example.

## Scope

Initial candidate APIs:

- `qk_rmsnorm_rope`
- `residual_rmsnorm_quant_nvfp4`
- `silu_mul_quant_nvfp4`
- `patch_embed_bias_pos`
- `patch_im2col`
- `video_conv_lowbit`
- `dit_norm_quant`
- `bf16_ncdhw_to_ndhwc_quant`

## Non-Goals

- Do not package complete VLA or video model pipelines.
- Do not expose model-specific scheduler or serving state.
- Do not include generic GEMM epilogues unless they are only meaningful for
  vision/video layouts.

## Baselines

Benchmarks should compare against PyTorch eager, FlashRT internal reference
paths, and a model-block baseline when the kernel replaces a known sequence of
ops.

## Showcase Criteria

- Avoid model-specific public names, but include benchmark labels for real VLA,
  vision-language, and video shapes.
- Prefer fused kernels that remove multiple launches or avoid intermediate
  layout conversions.
- Include a minimal HF-style Python example once a kernel can be used in a
  downstream model block.
