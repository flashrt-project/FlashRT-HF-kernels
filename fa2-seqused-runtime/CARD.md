---
tags:
  - kernel
  - cuda
  - flash-attention
  - inference
  - cuda-graphs
library_name: kernels
---

# FA2 Seqused Runtime

Allocation-free FlashAttention-2 forward operators for CUDA Graph inference.
The distinguishing feature is a device-resident per-batch `seqused_k`, allowing
one captured graph to serve changing valid K/V lengths without a host scalar
read or graph recapture.

## Available functions

- `forward(q, k, v, *, softmax_scale=None, causal=False, use_split_kv=True)`
- `forward_static(q, k, v, *, out, softmax_lse, workspace=None, softmax_scale=None, causal=False)`
- `forward_seqused_static(q, k, v, seqused_k, *, out, softmax_lse, workspace=None, softmax_scale=None)`
- `allocate_outputs(q)`
- `allocate_workspace(q, k, *, num_sms=None)`
- `recommended_num_splits(batch, seqlen_q, seqlen_k, heads_q, head_dim, num_sms)`
- `FA2Workspace`

## Example

```python
import torch
from kernels import get_kernel

fa2 = get_kernel("flashrt/fa2-seqused-runtime", version=1)
q = torch.randn(1, 16, 16, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 2048, 4, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
used = torch.tensor([1536], device="cuda", dtype=torch.int32)
out, lse = fa2.allocate_outputs(q)
workspace = fa2.allocate_workspace(q, k)

fa2.forward_seqused_static(
    q, k, v, used, out=out, softmax_lse=lse, workspace=workspace
)
```

The split-KV LSE reset is issued on the current stream and is captured with the
kernel. Updating `used` on device before replay changes the valid K/V length.

Causal calls use FlashAttention's bottom-right-aligned mask when query and KV
lengths differ. This is the chunked-prefill/verify convention, not PyTorch
SDPA's top-left `is_causal=True` convention for rectangular inputs.

## Scope

This package is forward-only and runtime-oriented. It intentionally does not
duplicate the complete training, varlen, backward, paged-cache, or dropout API
of `kernels-community/flash-attn2`.
