# Gated Delta Attention

FlashRT native CUDA BF16 Gated DeltaNet / linear-attention state kernels.

The first public profile targets Qwen3.6-style dimensions:

- value heads: `48`
- key/value head dim: `128`
- BF16 q/k/v/g/beta/out
- BF16 or FP32 recurrent state
- optional in-kernel Q/K L2 normalization

## Available Functions

- `gated_delta_recurrent_bf16(q, k, v, g, beta, state, use_qk_l2norm=True, out=None)`
- `gated_delta_recurrent_inout_bf16(q, k, v, g, beta, state_in, use_qk_l2norm=True, state_out=None, out=None)`
- `gated_delta_recurrent_f32state_bf16io(q, k, v, g, beta, state_f32, use_qk_l2norm=True, out=None)`
- `gated_delta_chunk_bf16(q, k, v, g, beta, state, use_qk_l2norm=True, out=None)`
- `gated_delta_chunk_smem_bf16(q, k, v, g, beta, state, use_qk_l2norm=True, out=None)`

The companion `flashrt/linear-attention-primitives` package provides split,
RoPE, and gating-preparation helpers. This package focuses on the stateful
linear-attention recurrence itself.

## Usage

```python
from kernels import get_kernel
import torch

gdn = get_kernel("flashrt/gated-delta-attention", version=1, trust_remote_code=True)

B, H, D = 1, 48, 128
q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
g = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
beta = torch.sigmoid(torch.randn(B, H, device="cuda")).to(torch.bfloat16)
state = torch.zeros(B, H, D, D, device="cuda", dtype=torch.bfloat16)

out = gdn.gated_delta_recurrent_bf16(q, k, v, g, beta, state)
```

## Validation

See `VALIDATION.md`.
