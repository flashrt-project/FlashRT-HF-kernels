# flashrt-vla-video

Reusable VLA, vision, video, and diffusion kernels from FlashRT.

This package should focus on gaps that are not already covered by common LLM
attention, MoE, and quantization packages.

The first buildable slice targets Q/K post-processing:

- Decode-time per-head RMSNorm, rotate-half RoPE, and staging/cache writes.
- Video/VLA packed-QKV split, Q/K RMSNorm, and interleaved RoPE.

These are launch-bound paths common in LLM, VLA, vision-language, and video
decoders.

## Scope

Implemented APIs:

- `q_norm_rope_bf16`
- `k_norm_rope_v_cache_bf16`
- `qkv_split_norm_rope_bf16`

Future candidate APIs:

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

The first internal benchmark shows the implemented Q/K decode post-processing
slice is the strongest current showcase candidate. Public benchmark tables
should be added after package-local validation.

## Showcase Criteria

- Avoid model-specific public names, but include benchmark labels for real VLA,
  vision-language, and video shapes.
- Prefer fused kernels that remove multiple launches or avoid intermediate
  layout conversions.
- Include a minimal HF-style Python example once a kernel can be used in a
  downstream model block.

## Usage

```python
import torch
from kernels import get_kernel

ops = get_kernel("flashrt/flashrt-vla-video", version=1, trust_remote_code=True)

q = torch.randn((48, 128), device="cuda", dtype=torch.bfloat16)
k = torch.randn((8, 128), device="cuda", dtype=torch.bfloat16)
v = torch.randn((8, 128), device="cuda", dtype=torch.bfloat16)
weight = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
cos = torch.randn((64,), device="cuda", dtype=torch.bfloat16)
sin = torch.randn((64,), device="cuda", dtype=torch.bfloat16)

q_stage = ops.q_norm_rope_bf16(q, weight, cos, sin)
k_cache, v_cache = ops.k_norm_rope_v_cache_bf16(k, v, weight, cos, sin)

packed_qkv = torch.randn((1, 256, 3 * 24 * 128), device="cuda", dtype=torch.bfloat16)
norm_q = torch.ones((24 * 128,), device="cuda", dtype=torch.bfloat16)
norm_k = torch.ones((24 * 128,), device="cuda", dtype=torch.bfloat16)
freqs_re = torch.randn((4096, 64), device="cuda", dtype=torch.float32)
freqs_im = torch.randn((4096, 64), device="cuda", dtype=torch.float32)

q_video, k_video = ops.qkv_split_norm_rope_bf16(
    packed_qkv, norm_q, norm_k, freqs_re, freqs_im, heads=24, head_dim=128
)
```
