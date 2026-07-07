# FlashRT INT8 Transformer Primitives

Tensor API package for model-neutral INT8 transformer building blocks.

## Available functions

```python
quantize_int8_static_bf16
quantize_int8_rowwise_bf16
quantize_int8_rowwise_static_bf16
rms_norm_quantize_int8_rowwise_bf16
residual_add_rms_norm_quantize_int8_rowwise_bf16
int8_rowwise_linear_bf16
int8_silu_gated_linear_bf16
```

## Usage

```python
from kernels import get_kernel

ops = get_kernel("flashrt/int8-transformer-primitives", version=1)
y = ops.int8_rowwise_linear_bf16(x_i8, w_i8, x_scale, w_scale)
```

## Notes

This is not a FlashRT serving-pointer API. It is a generic Tensor API intended
for direct use from PyTorch modules and framework integrations.
