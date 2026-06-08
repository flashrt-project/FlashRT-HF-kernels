# flashrt-residual-norm-quant

Tensor-facing FlashRT residual, RMSNorm, and static FP8 quantization kernels for
Hugging Face `kernels`.

This package is runtime glue for model hot paths that need to produce the next
layer's FP8 activation without falling back to multiple PyTorch operations:

```text
BF16 residual/x -> residual add -> RMSNorm -> static-scale FP8 E4M3 activation
```

## Exported APIs

- `rms_norm_bf16(x, weight, eps=1e-6, out=None)`
- `rms_norm_quant_fp8_static_bf16(x, weight, scale, eps=1e-6, out=None)`
- `residual_add_rms_norm_quant_fp8_static_bf16(residual, x, weight, scale, eps=1e-6, out=None)`

The residual API updates `residual` in place with `residual += x`, rounded to
BF16, then emits the normalized FP8 activation.

## Tensor Conventions

- `x`: BF16 tensor, shape `(rows, dim)`
- `residual`: BF16 tensor, shape `(rows, dim)`, in-place updated
- `weight`: BF16 tensor, shape `(dim,)`
- `scale`: CUDA `float32` scalar tensor used for static FP8 quantization
- `out`: FP8 E4M3 tensor, shape `(rows, dim)`

The hidden dimension must be even because this version uses FlashRT's packed
BF16 pair path.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel(
    "flashrt/flashrt-residual-norm-quant",
    version=1,
    trust_remote_code=True,
)

x = torch.randn((10, 1024), device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x)
weight = torch.ones((1024,), device="cuda", dtype=torch.bfloat16)
scale = torch.tensor([0.04], device="cuda", dtype=torch.float32)

x_fp8 = ops.residual_add_rms_norm_quant_fp8_static_bf16(
    residual,
    x,
    weight,
    scale,
    eps=1e-6,
)
```

## Validation

```bash
python flashrt-residual-norm-quant/tests/test_residual_norm_quant.py --backend source --mode full
python flashrt-residual-norm-quant/benchmarks/benchmark.py --backend source --shapes all
```

Current RTX 5090 source-extension rows pass with FP8 `p99_abs=0` across the
initial PI0.5/VLA shape grid. Built-artifact and multi-hardware validation are
pending.
