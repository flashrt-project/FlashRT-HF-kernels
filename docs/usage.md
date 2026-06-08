# FlashRT Kernel Hub Usage

This document is the public entry point for choosing and calling the FlashRT
Kernel Hub packages.

FlashRT kernels are published under the Hugging Face Kernel Hub namespace
`flashrt`. They are Tensor APIs for integration into PyTorch/Hugging Face code.
The full FlashRT model runtime and serving pipeline remain in
[LiangSu8899/FlashRT](https://github.com/LiangSu8899/FlashRT).

## Install And Load

Install the Hugging Face `kernels` package in the same Python environment as
PyTorch:

```bash
pip install kernels
```

Then load a package from the Hub:

```python
from kernels import get_kernel

ops = get_kernel("flashrt/flashrt-fp8-ffn", version=1, trust_remote_code=True)
```

The returned `ops` module exposes normal Python functions backed by compiled
CUDA extensions.

## Package Map

Published v1 packages:

| Package | What it contains | Use it when |
| --- | --- | --- |
| `flashrt/flashrt-fp8-ffn` | FP8 GEMM and full GELU FFN/MLP blocks. | You have pre-quantized FP8 activations and weights and want a reusable `FP8 up GEMM -> bias/GELU -> FP8 requant -> FP8 down GEMM -> bias` block. |
| `flashrt/flashrt-gemm-epilogues` | BF16 GEMM bias/GELU wrappers and BF16-to-FP8 quantization epilogues. | You need post-GEMM activation quantization, channel-scale quantization, or a small fused BF16 GEMM epilogue helper. |
| `flashrt/flashrt-vla-video` | VLA/video/diffusion Q/K and packed-QKV post-processing. | You need packed QKV split, Q/K RMSNorm, and RoPE staging for video or VLA attention blocks. |
| `flashrt/flashrt-nvfp4` | Blackwell NVFP4 scale-factor layout helpers. | You need CUTLASS Sm1xx-compatible NVFP4 swizzled scale-factor buffers. |
| `flashrt/flashrt-smallm-gemm` | Shape-specialized SM120 NVFP4 W4A4 decode matvec. | You need a low-batch/decode M=1 W4A4 matvec with BF16 output on supported Blackwell shapes. |
| `flashrt/flashrt-fused-quant` | Fused activation plus NVFP4 quantization. | You need memory-bound `SiLU(gate) * up` activation and NVFP4 swizzled quantization in one call. |

Source-validated V2 candidate packages for the VLA/world-model runtime demo:

| Package | What it contains | Use it when |
| --- | --- | --- |
| `flashrt/flashrt-fp8-swiglu-ffn` | FP8 gate/up GEMM, SiLU product, FP8 requant, and FP8 down GEMM for SwiGLU FFNs. | You want Gemma-style VLA/VLM language-path FFN islands without returning to PyTorch between FP8 GEMMs. |
| `flashrt/flashrt-residual-norm-quant` | Residual add, RMSNorm, and static FP8 activation producer kernels. | You need to feed adjacent FP8 blocks from a BF16 residual path with one fused producer. |
| `flashrt/flashrt-qkv-cache-rope` | Packed-QKV split, Q/K RMSNorm, RoPE, joint Q/K/V workspace writes, decode Q staging, and KV cache-write. | You need attention staging for VLA/VLM/video blocks, including graph-friendly single-token decode cache writes. |
| `flashrt/flashrt-vla-residual-gates` | Video/action/und joint gated residual updates. | You have multi-segment VLA block glue and want one CUDA launch for the segment residual/gate updates. |
| `flashrt/flashrt-adaptive-norms` | AdaRMSNorm/style modulation and fused residual/AdaRMSNorm/static-FP8 output. | You need DiT/VLA/world-model adaptive normalization and optional FP8 activation output. |
| `flashrt/flashrt-spatiotemporal-layout` | NCDHW/BLC layout, temporal unshuffle, channel-bias, and short-cache helpers. | You need world-model/video layout glue to keep the model-demo hot path on CUDA. |

The V2 candidates are not public hardware claims until their kernel-builder
artifacts and installed-artifact validation are completed.

## Quick Examples

### Full FP8 GELU FFN

```python
from kernels import get_kernel

ops = get_kernel("flashrt/flashrt-fp8-ffn", version=1, trust_remote_code=True)

y = ops.fp8_gelu_mlp_bf16(
    x_fp8,          # (M, K), torch.float8_e4m3fn
    up_w_fp8,       # (H, K), torch.float8_e4m3fn
    up_bias,        # (H,), torch.bfloat16
    down_w_fp8,     # (N, H), torch.float8_e4m3fn
    down_bias,      # (N,), torch.bfloat16
    x_scale,        # CUDA float32 scalar
    up_w_scale,     # CUDA float32 scalar
    hidden_scale,   # CUDA float32 scalar
    down_w_scale,   # CUDA float32 scalar
)
```

Run a complete minimal script:

```bash
python examples/minimal_fp8_ffn.py
```

Run a model-integration skeleton that replaces a PyTorch
`Linear -> GELU(tanh) -> Linear` module:

```bash
python examples/replace_torch_ffn.py
```

### Packed QKV Postprocess For Video/VLA Blocks

```python
from kernels import get_kernel

ops = get_kernel("flashrt/flashrt-vla-video", version=1, trust_remote_code=True)

q, k = ops.qkv_split_norm_rope_bf16(
    packed_qkv,       # (batch, tokens, 3 * heads * head_dim), BF16
    norm_q_weight,    # (heads * head_dim,), BF16
    norm_k_weight,    # (heads * head_dim,), BF16
    freqs_re,         # (rope_table_len, head_dim / 2), FP32
    freqs_im,         # (rope_table_len, head_dim / 2), FP32
    heads=24,
    head_dim=128,
)
```

### VLA Runtime QKV And Decode Cache Path

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/flashrt-qkv-cache-rope", version=1, trust_remote_code=True)

q_cat, k_cat, v_cat = ops.qkv_split_joint3_cat_bf16(
    packed_v,
    qkv_v_bias,
    norm_v_q_weight,
    norm_v_k_weight,
    freqs_re,
    freqs_im,
    packed_a,
    norm_a_q_weight,
    norm_a_k_weight,
    packed_u,
    norm_u_q_weight,
    norm_u_k_weight,
    heads=24,
    head_dim=128,
    q_cat_out=q_cat,
    k_cat_out=k_cat,
    v_cat_out=v_cat,
)

q_buf = ops.decode_q_norm_rope_stage_bf16(q_pre, q_norm_weight, cos, sin)
ops.decode_k_norm_rope_kvwrite_devpos_bf16(
    k_pre,
    v_pre,
    k_norm_weight,
    cos,
    sin,
    cur_pos_int32_cuda,
    k_cache,
    v_cache,
)
```

`decode_*` APIs are fixed to `head_dim == 128` and use BF16 `(64,)` cos/sin
vectors with a rotate-half RoPE contract. Unsupported shapes are rejected.

### World-Model Layout Glue

```python
from kernels import get_kernel

ops = get_kernel("flashrt/flashrt-spatiotemporal-layout", version=1, trust_remote_code=True)

tokens = ops.ncdhw_to_blc_bf16(latents_ncdhw)
expanded = ops.time_unshuffle2_bf16(two_channel_latents)
ops.add_bias_ncdhw_bf16(latents_ncdhw, channel_bias)
cache = ops.update_cache2_ncdhw_bf16(cur_latent, prev_cache)
```

### BF16 To FP8 Quantization Epilogue

```python
from kernels import get_kernel

ops = get_kernel("flashrt/flashrt-gemm-epilogues", version=1, trust_remote_code=True)

out_fp8 = ops.bias_gelu_quantize_fp8_static_bf16(
    hidden_bf16,
    bias_bf16,
    output_scale,
)
```

### NVFP4 Scale Layout

```python
from kernels import get_kernel

ops = get_kernel("flashrt/flashrt-nvfp4", version=1, trust_remote_code=True)

swizzled = ops.nvfp4_sf_linear_to_swizzled(linear_scale_bytes)
```

### Fused SiLU Product Plus NVFP4 Quantization

```python
from kernels import get_kernel

ops = get_kernel("flashrt/flashrt-fused-quant", version=1, trust_remote_code=True)

packed, scales = ops.silu_mul_quant_nvfp4_swizzled_bf16(gate_bf16, up_bf16)
```

## Model Integration Rules

Use the kernels as continuous blocks, not as tiny Python calls sprinkled between
unfused PyTorch operations.

For FP8 FFN integration:

- Keep weights pre-quantized and store scales as buffers.
- Calibrate activation and hidden scales before benchmarking.
- Preallocate scratch buffers for repeated shapes.
- Prefer passing FP8 activations directly between FlashRT blocks.
- Avoid repeatedly converting BF16 to FP8 at every layer unless that conversion
  is part of the kernel path being measured.

For video/VLA attention integration:

- Replace the complete QKV postprocess island: packed QKV split, Q/K RMSNorm,
  and RoPE.
- For decode paths, replace the complete Q/K RMSNorm + rotate-half RoPE +
  Q staging/KV cache-write island. Prefer the device-position KV-write API
  when the loop is CUDA Graph captured.
- Keep the same attention implementation on both baseline and FlashRT paths if
  the goal is attribution for the postprocess kernel.

For VLA/world-model demo integration:

- Keep package calls as continuous model islands: QKV workspace write, decode
  cache write, residual/gate update, adaptive norm, and spatiotemporal layout.
- Preallocate output and cache tensors. Avoid allocating inside the timed loop.
- Use the official PyTorch model path as the baseline, not the FlashRT serving
  runtime, when the goal is ecosystem-facing acceleration.

For model-level benchmarks:

- Compare against the model's official PyTorch/eager path, not against the
  already-optimized FlashRT serving runtime.
- Warm up both paths and synchronize CUDA timers.
- Report correctness together with speed: max error, mean error, p99 error or
  cosine similarity, dtype, tolerance, shape, hardware, driver, CUDA, PyTorch,
  and package version.

## Torch Compile

The Python wrappers register fake/meta kernels so these operators can appear in
`torch.compile` graphs. A typical call looks like:

```python
compiled = torch.compile(ops.fp8_gelu_mlp_bf16, fullgraph=True)
y = compiled(x_fp8, up_w_fp8, up_bias, down_w_fp8, down_bias, x_s, up_s, h_s, d_s)
```

For fair benchmark baselines, verify that the compiled PyTorch reference is
numerically equivalent to the eager PyTorch reference before reporting
`vs torch.compile`. FP8 fake-quant chains can be sensitive at rounding
boundaries, so FlashRT benchmarks use compile-stable references where needed.

Run the repository smoke check:

```bash
python scripts/torch_compile_smoke.py --version 1
```

## Current Hardware Scope

Version 1 packages have been validated locally on RTX 5090 with the
`torch211-cxx11-cu128-x86_64-linux` artifact path. Additional hardware
validation is in progress. Package cards and benchmark tables should not be
read as broader hardware claims until the corresponding hardware rows are
published.
