# transformer-layout-primitives

Generic BF16 layout, RoPE, and text-state primitives for transformer pipelines.

Hub repo: `flashrt/transformer-layout-primitives`

## Public API

- `fill_neginf_bf16(dst) -> dst`
- `add_bias_bf16_(data, bias) -> data`
- `repeat_interleave_heads_bf16(src, repeat) -> bf16`
- `text_gather_bf16(src, batch, seq) -> bf16`
- `text_scatter_bf16(dst, src, batch, seq) -> dst`
- `rope_rotate_half_bf16_(x, cos, sin) -> x`
- `qk_rmsnorm_rope_bf16_(qk, weight, cos, sin, eps=1e-6) -> qk`

The package is intentionally model-neutral. It exposes Tensor APIs for common
transformer integration gaps: head repeat for GQA/MQA, first/last token gather
and scatter, bias add, RoPE rotate-half, and fused Q/K RMSNorm+RoPE.

## Example

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/transformer-layout-primitives", version=1)

q = torch.randn((128, 32, 128), device="cuda", dtype=torch.bfloat16)
weight = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
cos = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
sin = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
ops.qk_rmsnorm_rope_bf16_(q, weight, cos, sin)
```

## Shape contract

- All tensors are contiguous CUDA BF16 tensors.
- `repeat_interleave_heads_bf16`: `src` is `(seq, heads, head_dim)`.
- `text_gather_bf16`: `src` is flattened `(batch * seq, dim)` and returns
  first and last token rows as `(2 * batch, dim)`.
- `text_scatter_bf16`: writes `(2 * batch, dim)` rows back to first and last
  positions in `(batch * seq, dim)`.
- RoPE functions use rotate-half layout with `cos/sin` shaped `(seq, head_dim)`
  or `(rows, head_dim)`.

## Validation

Correctness is tested against PyTorch BF16/FP32 reference formulas with exact
checks for pure layout operations and strict BF16 tolerances for math ops.

See `benchmarks/RESULTS.md` for current local RTX 5090 source benchmark data.
