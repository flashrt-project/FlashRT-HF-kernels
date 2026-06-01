# flashrt-vla-video

Draft package for reusable VLA, vision, video, and diffusion kernels from
FlashRT.

This package should focus on gaps that are not already covered by common LLM
attention, MoE, and quantization packages.

## Scope

Initial candidate APIs:

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

Benchmarks should compare against PyTorch eager and FlashRT internal reference
paths.
