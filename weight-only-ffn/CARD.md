---
library_name: kernels
license: apache-2.0
tags:
  - cuda
  - native-cuda
  - flashrt
  - blackwell
  - weight-only
  - int8
  - nvfp4
  - transformers
---

# weight-only-ffn

Small-M BF16-activation FFN regions with static W4 or W8 weights for Blackwell
SM120/SM121.

Available functions:

- `quantize_w4_weight_bf16`
- `dequantize_w4_weight_bf16`
- `quantize_w8_weight_bf16`
- `dequantize_w8_weight_bf16`
- `w4a16_linear_bf16`
- `w8a16_linear_bf16`
- `w4a16_swiglu_ffn_bf16`
- `w4a16_geglu_ffn_bf16`
- `w4a16_gelu_ffn_bf16`
- `w8a16_swiglu_ffn_bf16`
- `w8a16_geglu_ffn_bf16`
- `w8a16_gelu_ffn_bf16`

```python
from kernels import get_kernel

ops = get_kernel(
    "flashrt/weight-only-ffn",
    version=1,
    trust_remote_code=True,
)
packed, scales = ops.quantize_w8_weight_bf16(weight_bf16)
y = ops.w8a16_linear_bf16(x_bf16, packed, scales)
```

Production auto dispatch supports qualified `M=1..4` shapes and rejects known
slow regions based on row count and weight dimensions.
Weights are prepared once; activations remain BF16 throughout the public
contract.
