# flashrt-vla-video

Reusable VLA, vision, video, and diffusion attention postprocess kernels from
FlashRT.

This package focuses on gaps that are not already covered by common LLM
attention, MoE, and quantization packages. It remains a normal maintained
FlashRT Kernel Hub package. If the public namespace needs to be reorganized in
the future, that should be handled as an explicit migration plan.

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

Future VLA/world-model APIs should be added here only when they match the
package scope. More specialized runtime glue can live in focused packages such
as `flashrt-qkv-cache-rope` or `flashrt-spatiotemporal-layout`.

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

See `examples/qkv_postprocess_block.py` for a minimal HF-style module that
replaces packed-QKV split, Q/K RMSNorm, and interleaved RoPE with one fused
kernel call.

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

For the newer focused QKV/cache package:

```python
ops = get_kernel("flashrt/flashrt-qkv-cache-rope", version=1, trust_remote_code=True)
```
