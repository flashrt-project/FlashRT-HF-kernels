# flashrt-vla-residual-gates

Tensor-facing FlashRT VLA joint residual/gate kernels for Hugging Face
`kernels`.

This package targets the post-attention and post-MLP glue in VLA/video blocks
where three token groups are updated together:

```text
video:  residual + (x + bias) * gate
action: residual + (x + bias) * gate    or residual + x * gate
und:    residual + x
```

The fused API avoids three separate PyTorch elementwise chains and writes the
three BF16 outputs in one CUDA launch.

## Exported APIs

- `joint3_bias_gate_residual_bf16(v_residual, v_x, v_bias, v_gate, a_residual, a_x, a_bias, a_gate, u_residual, u_x, v_out=None, a_out=None, u_out=None)`
- `joint3_bias_gate_residual_action_nobias_bf16(v_residual, v_x, v_bias, v_gate, a_residual, a_x, a_gate, u_residual, u_x, v_out=None, a_out=None, u_out=None)`

## Tensor Conventions

- Segment tensors are contiguous BF16 matrices with shape `(rows, dim)`.
- Bias tensors are contiguous BF16 vectors with shape `(dim,)`.
- `dim` must be even.
- Outputs are BF16 matrices with the same shape as their segment residual.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel(
    "flashrt/flashrt-vla-residual-gates",
    version=1,
    trust_remote_code=True,
)

dim = 3072
v_residual = torch.randn((2520, dim), device="cuda", dtype=torch.bfloat16)
v_x = torch.randn_like(v_residual)
v_gate = torch.randn_like(v_residual)
v_bias = torch.zeros((dim,), device="cuda", dtype=torch.bfloat16)

a_residual = torch.randn((16, dim), device="cuda", dtype=torch.bfloat16)
a_x = torch.randn_like(a_residual)
a_gate = torch.randn_like(a_residual)

u_residual = torch.randn((16, dim), device="cuda", dtype=torch.bfloat16)
u_x = torch.randn_like(u_residual)

v_out, a_out, u_out = ops.joint3_bias_gate_residual_action_nobias_bf16(
    v_residual,
    v_x,
    v_bias,
    v_gate,
    a_residual,
    a_x,
    a_gate,
    u_residual,
    u_x,
)
```

## Validation

```bash
python flashrt-vla-residual-gates/tests/test_vla_residual_gates.py --backend source --mode full
python flashrt-vla-residual-gates/benchmarks/benchmark.py --backend source --shapes all
```

Current RTX 5090 source-extension validation is bit-level exact against the
PyTorch eager reference across the source shape grid. Built-artifact and
multi-hardware validation are pending.
