# flashrt-gemm-epilogues

First buildable package for FlashRT fused GEMM and quantization epilogue
kernels.

This package should expose generic Linear/GEMM-style APIs with fused output
work, rather than raw FlashRT internal launchers. The first target is a small
set of operations that remove common post-GEMM launches in Transformer, VLA,
and diffusion inference.

The strongest current surface is FP8 quantization epilogue fusion:

- BF16 input plus optional BF16 bias.
- GELU(tanh) or per-channel scaling.
- Direct FP8 e4m3 output.

The BF16 GEMM epilogue wrappers are included for completeness and selected
decode shapes, but they are shape-sensitive and should not be used as the main
performance headline without target-shape validation.

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
``per-channel scale + FP8 quantize`` pattern. The full GEMM wrappers cover
``BF16 GEMM + BF16 bias -> BF16`` and
``BF16 GEMM + BF16 bias + GELU -> BF16`` using cuBLASLt epilogue support.

## Non-Goals

- Do not expose raw `cutlass_*` tuning variant functions as the main public API.
- Do not expose pointer-only APIs.
- Do not encode model names in function names.
- Do not depend on FlashRT runtime contexts.

## Baselines

Benchmarks should compare against:

- `torch.addmm` plus PyTorch elementwise epilogue for GEMM paths.
- `torch.nn.functional.linear` plus PyTorch elementwise epilogue for model-level
  checks.
- Existing FlashRT pybind path for internal regression checks.
- cuBLASLt/CUTLASS unfused path when available.

The BF16 GEMM epilogue wrapper is shape-sensitive on the current local RTX 5090
environment. The FP8 quantize epilogue kernels are this package's strongest
v1 surface; GEMM epilogue shapes should be promoted only when they beat the
stricter `torch.addmm` baseline.

## V1 Role

This package is the FP8/GEMM epilogue block in the v1 batch. Public messaging
should emphasize the FP8 quantization epilogue helpers; BF16 GEMM epilogue
results should be presented per shape.

## Validation

HF builder validation has passed for the torch211 CUDA 12.8, CUDA 12.6, and
CUDA 13.0 variants. Host-side CUDA correctness smoke has passed on the local
RTX 5090 environment. See `VALIDATION.md` for exact variants, commands,
environment, and known gaps.

See `examples/fp8_quant_epilogue_block.py` for a minimal HF-style module that
replaces BF16 post-projection epilogue work with the FP8 quantization helpers.

## Usage

```python
import torch
from kernels import get_kernel

ops = get_kernel("flashrt/flashrt-gemm-epilogues", version=1)

x = torch.randn((4, 4096), device="cuda", dtype=torch.bfloat16)
bias = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)
scale = torch.tensor([0.25], device="cuda", dtype=torch.float32)

y_fp8 = ops.bias_gelu_quantize_fp8_static_bf16(x, bias, scale)

channel_scale = torch.ones((4096,), device="cuda", dtype=torch.bfloat16)
y_scaled_fp8 = ops.channel_scale_quantize_fp8_static_bf16(x, channel_scale, scale)

a = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
gemm_bias = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)

y_gemm = ops.bf16_gemm_bias_gelu(a, b, gemm_bias)
```
