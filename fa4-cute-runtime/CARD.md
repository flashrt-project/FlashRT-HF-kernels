---
library_name: kernels
license: bsd-3-clause
tags: [cuda, blackwell, flash-attention, cute-dsl, inference]
---

# flashrt/fa4-cute-runtime

Forward-only FlashAttention-4 CuTe DSL runtime used by FlashRT's GROOT N1.7
and PI0.5 Thor pipelines. The source is vendored under the private `flashrt_fa4`
namespace and does not shadow an installed `flash_attn` package.

This package adds the dedicated Blackwell D256 2CTA forward path required by
PI0.5's 8-Q/1-KV-head PaliGemma encoder. The community FlashAttention-4 package
already covers the D48/D72/D128 family; it does not currently expose D256 on
SM100/SM110.

## Functions

- `flash_attn_func`
- `flash_attn_varlen_func`
- `forward_static`

```python
from kernels import get_kernel

fa4 = get_kernel("flashrt/fa4-cute-runtime", version=1)
out = fa4.flash_attn_func(q, k, v, causal=False)
```

The vendored forward wrapper returns `(out, lse)`; use `result[0]` when only
the attention output is needed.

For a CUDA Graph hot path, preallocate the output and use the allocation-free
entry point:

```python
out = torch.empty_like(q)
fa4.forward_static(q, k, v, out, causal=False)
```

For a padded fixed-shape graph, pass the valid K/V length as a CUDA int32
tensor. PI0.5 uses this form for its encoder cache:

```python
seqused_k = torch.tensor([valid_k], device="cuda", dtype=torch.int32)
fa4.forward_static(
    q, k_padded, v_padded, out,
    causal=False,
    pack_gqa=True,
    seqused_k=seqused_k,
)
```

Inputs follow FlashAttention's `(batch, sequence, heads, head_dim)` contract.
Qualified model profiles include D72 MHA and D256 GQA (8 Q heads / 1 KV head),
with both dense and `seqused_k` execution. This package targets SM100-family
Blackwell forward inference and requires
CUDA 13 plus `nvidia-cutlass-dsl` 4.4.x or 4.5.x. The wrapper selects Thor's
accepted architecture alias according to the installed DSL version.

This is an execution backend rather than a universal SDPA replacement. Select
it with model-shape profiling; the GROOT causal GQA profile benefits while some
short vision profiles remain faster on PyTorch SDPA.
