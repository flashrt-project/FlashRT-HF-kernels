# FlashRT Blockwise FP8 Producers

Fused normalization and activation producers for block-128 FP8 transformer
GEMMs.

## Available functions

```text
quantize_fp8_block128_bf16
layer_norm_fp8_block128_bf16
rms_norm_fp8_block128_bf16
residual_add_rms_norm_fp8_block128_bf16
gelu_tanh_fp8_block128_bf16
gelu_tanh_bias_fp8_block128_bf16
silu_mul_fp8_block128_bf16
silu_mul_merged_fp8_block128_bf16
```

## Example

```python
from kernels import get_kernel

ops = get_kernel(
    "flashrt/blockwise-fp8-producers",
    version=1,
    trust_remote_code=True,
)
fp8_hidden, scale = ops.gelu_tanh_bias_fp8_block128_bf16(hidden, bias)
```

Inputs are contiguous CUDA BF16 tensors. Outputs use FP8 E4M3 plus FP32
per-row/per-128-channel scales. See the README for full signatures and shape
requirements.
