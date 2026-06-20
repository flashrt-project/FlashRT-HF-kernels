# Causal Conv1D State

FlashRT native CUDA BF16 causal depthwise Conv1D kernels for transformer
decode, verify, and prefill state updates.

The first public profile targets Qwen3.6-style linear-attention blocks:
`conv_dim=10240`, `kernel_size=4`, BF16 inputs/weights/state, fused optional
SiLU, and state layout `(B, C, K - 1)`. The kernels are exposed under generic
names so they can be reused by other stateful Conv1D transformer runtimes.

## Available Functions

- `causal_conv1d_bf16(x, w, bias=None, apply_silu=True, out=None)`
- `causal_conv1d_update_bf16(x_new, w, state, bias=None, apply_silu=True, out=None)`
- `causal_conv1d_update_inout_bf16(x_new, w, state_in, bias=None, apply_silu=True, out=None, state_out=None)`
- `causal_conv1d_update_chunk_bf16(x, w, state, bias=None, apply_silu=True, out=None)`
- `causal_conv1d_update_chunk_parallel_bf16(x, w, state, bias=None, apply_silu=True, out=None)`
- `causal_conv1d_update_chunk_parallel_gqa_bf16(x, w, state, bias=None, apply_silu=True, q16=None, k16=None, v48=None)`

Unsupported shapes fail at the boundary rather than falling back silently.

## Usage

```python
from kernels import get_kernel
import torch

conv = get_kernel("flashrt/causal-conv1d-state", version=1, trust_remote_code=True)

B, S, C, K = 1, 8, 10240, 4
x = torch.randn(B, S, C, device="cuda", dtype=torch.bfloat16)
w = torch.randn(C, K, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(C, device="cuda", dtype=torch.bfloat16)
state = torch.zeros(B, C, K - 1, device="cuda", dtype=torch.bfloat16)

out = conv.causal_conv1d_update_chunk_parallel_bf16(x, w, state, bias)
```

For static decode, preallocate `out` and use `update_inout` to avoid hidden
state copies:

```python
x_new = x[:, 0]
state_next = torch.empty_like(state)
out = torch.empty_like(x_new)
conv.causal_conv1d_update_inout_bf16(
    x_new, w, state, bias, out=out, state_out=state_next
)
```

## Validation

See `VALIDATION.md` and `benchmarks/RESULTS.md`.
