# FlashRT Transformer Layout Primitives

Tensor API package for model-neutral transformer layout and RoPE helpers.

## Available functions

```python
fill_neginf_bf16
add_bias_bf16_
repeat_interleave_heads_bf16
text_gather_bf16
text_scatter_bf16
rope_rotate_half_bf16_
qk_rmsnorm_rope_bf16_
```

## Usage

```python
from kernels import get_kernel

ops = get_kernel("flashrt/transformer-layout-primitives", version=1)
q = ops.qk_rmsnorm_rope_bf16_(q, weight, cos, sin)
```

## Notes

This is not a FlashRT serving-pointer API. It is a generic Tensor API intended
for direct use from PyTorch modules, Transformers integrations, and model
runtime prototypes.
