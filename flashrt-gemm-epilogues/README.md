# flashrt-gemm-epilogues

First buildable package for FlashRT fused GEMM epilogue kernels.

This package should expose generic Linear/GEMM-style APIs with fused output
work, rather than raw FlashRT internal launchers. The first target is a small
set of operations that remove common post-GEMM launches in Transformer, VLA,
and diffusion inference.

## Scope

Implemented APIs:

- `bf16_gemm_bias`
- `bf16_gemm_bias_gelu`
- `bias_gelu_quantize_fp8_static_bf16`
- `channel_scale_quantize_fp8_static_bf16`
- `gelu_quantize_fp8_static_bf16`

Planned APIs:

- `fp8_linear_bias`
- `fp8_linear_bias_gelu`
- `fp8_linear_bias_silu`
- `fp8_linear_bias_residual`
- `fp8_linear_quant_out`
- `nvfp4_linear_bias`
- `nvfp4_linear_bias_gelu`
- `nvfp4_linear_quant_out`

The first implemented kernels cover the post-GEMM epilogue pattern
``bias + GELU(tanh) + FP8 quantize`` and the adjacent
``per-channel scale + FP8 quantize`` pattern. The first full GEMM wrapper is
``BF16 GEMM + BF16 bias + GELU -> BF16`` using cuBLASLt epilogue support.

## Non-Goals

- Do not expose raw `cutlass_*` tuning variant functions as the main public API.
- Do not expose pointer-only APIs.
- Do not encode model names in function names.
- Do not depend on FlashRT runtime contexts.

## Baselines

Benchmarks should compare against:

- `torch.nn.functional.linear` plus PyTorch elementwise epilogue.
- Existing FlashRT pybind path for internal regression checks.
- cuBLASLt/CUTLASS unfused path when available.

## Promotion Target

This is the recommended first buildable package.

## Validation

HF builder validation has passed for the torch211 CUDA 12.8 and CUDA 12.6
variants. Host-side CUDA correctness smoke has passed on the local RTX 5090
environment. See `VALIDATION.md` for exact variants, commands, environment,
and known gaps.

## Usage

```python
import torch
from kernels import get_kernel

ops = get_kernel("flashrt/flashrt-gemm-epilogues", version=1, trust_remote_code=True)

a = torch.randn((64, 4096), device="cuda", dtype=torch.bfloat16)
b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
bias = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)

y = ops.bf16_gemm_bias_gelu(a, b, bias)
y_no_activation = ops.bf16_gemm_bias(a, b, bias)

x = torch.randn((4, 4096), device="cuda", dtype=torch.bfloat16)
bias = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)
scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)

y_fp8 = ops.bias_gelu_quantize_fp8_static_bf16(x, bias, scale)

channel_scale = torch.ones((4096,), device="cuda", dtype=torch.bfloat16)
y_scaled_fp8 = ops.channel_scale_quantize_fp8_static_bf16(x, channel_scale, scale)
```
