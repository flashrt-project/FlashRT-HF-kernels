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
SM110/SM120/SM121.

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

Production auto dispatch keeps W4 in its qualified `M=1..4` domain. W8 linear
supports the qualified `M=1..8` domain, including the `K=17408` draft
down-projection envelope, and rejects unsupported rows explicitly.
Weights are prepared once; activations remain BF16 throughout the public
contract.

SM110 uses an independent CUDA 13 component and architecture-specific
performance gates. Standalone Thor linears are accepted only for sufficiently
large wide projections; callers should use the complete FFN API where the
whole region is qualified.
