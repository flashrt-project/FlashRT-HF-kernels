# flashrt/sageattention2-blackwell

FlashRT SageAttention2-style Blackwell prefill attention kernels.

Use this package for long-context prefill/self-attention where Q/K are quantized
to int8 and V is kept in FP16 or FP8 Sage layout. It supports:

- Wan/video non-causal self-attention.
- Qwen-style causal prefill.
- GQA, including `32/8` heads with `head_dim=128`.

This is not a decode attention kernel. For decode over FP8 K/V cache, use
`flashrt/fp8-kv-attention`.

## Quick Start

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/sageattention2-blackwell", version=1, trust_remote_code=True)

q = torch.randn((1, 4096, 32, 128), device="cuda", dtype=torch.bfloat16)
k = torch.randn((1, 4096, 8, 128), device="cuda", dtype=torch.bfloat16)
v = torch.randn((1, 4096, 8, 128), device="cuda", dtype=torch.bfloat16)

out = ops.sage2_prefill_f16_bf16_d128(q, k, v, causal=True)
```

See `README.md` for all functions, tensor contracts, and validation details.
