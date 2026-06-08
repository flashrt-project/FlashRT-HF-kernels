# flashrt-adaptive-norms

Tensor-facing FlashRT adaptive normalization kernels for Hugging Face `kernels`.

This package targets DiT, VLA, video, and world-model blocks that combine
RMSNorm with per-row style scale/shift/gate parameters:

```text
AdaRMSNorm: x -> RMSNorm(x, weight) -> norm * (1 + style_scale) + style_shift
Fused gate path: residual += x * gate -> AdaRMSNorm(residual) -> static FP8
```

## Exported APIs

- `ada_rms_norm_style_bf16(x, weight, style, eps=1e-6, out=None, gate_out=None)`
- `gate_residual_ada_norm_fp8_static_bf16(residual, x, gate, weight, style, scale, eps=1e-6, out=None, gate_out=None)`

## Tensor Conventions

- `x`, `residual`, `gate`: contiguous BF16 matrices, shape `(rows, dim)`
- `weight`: contiguous BF16 vector, shape `(dim,)`
- `style`: contiguous BF16 matrix, shape `(rows, 3 * dim)`
  - first `dim`: style scale
  - second `dim`: style shift
  - third `dim`: gate output
- `scale`: CUDA FP32 scalar tensor for static FP8 quantization
- FP8 output dtype: `torch.float8_e4m3fn`
- `dim` must be even.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel(
    "flashrt/flashrt-adaptive-norms",
    version=1,
    trust_remote_code=True,
)

rows, dim = 2520, 3072
x = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x)
gate = torch.randn_like(x)
weight = torch.ones((dim,), device="cuda", dtype=torch.bfloat16)
style = torch.randn((rows, 3 * dim), device="cuda", dtype=torch.bfloat16)
scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)

out_bf16, gate_out = ops.ada_rms_norm_style_bf16(x, weight, style)
residual, out_fp8, gate_out = ops.gate_residual_ada_norm_fp8_static_bf16(
    residual,
    x,
    gate,
    weight,
    style,
    scale,
)
```

## Validation

```bash
python flashrt-adaptive-norms/tests/test_adaptive_norms.py --backend source --mode full
python flashrt-adaptive-norms/benchmarks/benchmark.py --backend source --shapes all
```

Current RTX 5090 source-extension validation passes the full source grid.
Residual and gate outputs are bit-level exact. FP8 output uses a boundary-aware
gate because PyTorch and CUDA FP8 casters can choose adjacent FP8 values for
rare tie cases; p99_abs is zero across the source grid. Built-artifact and
multi-hardware validation are pending.
