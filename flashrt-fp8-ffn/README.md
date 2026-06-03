# flashrt-fp8-ffn

Tensor-facing FlashRT FP8 GEMM and FFN blocks for Hugging Face `kernels`.

This package exports reusable FP8 FFN surfaces from the FlashRT serving stack
without exposing raw pointer APIs:

- `fp8_gemm_bf16`: per-tensor FP8 E4M3 GEMM with BF16 output.
- `fp8_linear_bias_gelu_quant_bf16`: FP8 GEMM, BF16 bias/GELU, and FP8 output
  quantization.
- `fp8_gelu_mlp_bf16`: complete GELU MLP block:
  `FP8 up GEMM -> bias/GELU -> FP8 quant -> FP8 down GEMM -> bias`.

The API is generic. PI0.5/GROOT/Wan-shaped demos live outside the package.

## Scope

Inputs and weights are row-major FP8 E4M3 tensors:

- activation/input: `(M, K)`
- weight: `(N, K)`
- output: `(M, N)` BF16
- scale tensors: CUDA float32 scalars

The first version uses per-tensor scales because this mirrors the production
FlashRT FP8 descale paths used by the current PI0.5/GROOT RTX frontends. Native
block-scaled FP8 and shape-locked megakernels should be separate follow-up
packages once their Tensor API and hardware scope are finalized.

## Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/flashrt-fp8-ffn", version=1, trust_remote_code=True)

x = torch.randn((512, 1024), device="cuda", dtype=torch.bfloat16)
w_up = torch.randn((4096, 1024), device="cuda", dtype=torch.bfloat16)
w_dn = torch.randn((1024, 4096), device="cuda", dtype=torch.bfloat16)

x_scale = torch.tensor([0.05], device="cuda")
up_scale = torch.tensor([0.04], device="cuda")
hidden_scale = torch.tensor([0.25], device="cuda")
dn_scale = torch.tensor([0.04], device="cuda")

x_fp8 = torch.clamp(x.float() / x_scale, -448, 448).to(torch.float8_e4m3fn)
w_up_fp8 = torch.clamp(w_up.float() / up_scale, -448, 448).to(torch.float8_e4m3fn)
w_dn_fp8 = torch.clamp(w_dn.float() / dn_scale, -448, 448).to(torch.float8_e4m3fn)

y = ops.fp8_gelu_mlp_bf16(
    x_fp8,
    w_up_fp8,
    torch.zeros((4096,), device="cuda", dtype=torch.bfloat16),
    w_dn_fp8,
    torch.zeros((1024,), device="cuda", dtype=torch.bfloat16),
    x_scale,
    up_scale,
    hidden_scale,
    dn_scale,
)
```

## Validation

Run from the repository root:

```bash
python flashrt-fp8-ffn/tests/test_fp8_ffn.py --backend source
python flashrt-fp8-ffn/benchmarks/benchmark.py --backend source --shapes headline --compile-baseline
python flashrt-fp8-ffn/benchmarks/benchmark.py --backend source --shapes all
```

The package benchmark should be reported against PyTorch eager and
`torch.compile` references for headline rows. Model-block demos should be
reported separately from full model generation throughput claims.
