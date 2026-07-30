# blockwise-fp8-producers

Generic fused BF16 producers for per-token, per-128-channel FP8 E4M3
activations.

Hub repo: `flashrt/blockwise-fp8-producers`

## Public API

- `quantize_fp8_block128_bf16`
- `layer_norm_fp8_block128_bf16`
- `rms_norm_fp8_block128_bf16`
- `residual_add_rms_norm_fp8_block128_bf16`
- `gelu_tanh_fp8_block128_bf16`
- `gelu_tanh_bias_fp8_block128_bf16`
- `silu_mul_fp8_block128_bf16`
- `silu_mul_merged_fp8_block128_bf16`

Every function returns FP8 values shaped `(rows, dim)` and FP32 scales shaped
`(rows, dim / 128)`. The residual function additionally returns the BF16
residual sum.

## Usage

```python
import torch
from kernels import get_kernel

ops = get_kernel(
    "flashrt/blockwise-fp8-producers",
    version=1,
    trust_remote_code=True,
)

x = torch.randn((51, 4096), device="cuda", dtype=torch.bfloat16)
weight = torch.ones((4096,), device="cuda", dtype=torch.bfloat16)
fp8_x, block_scale = ops.rms_norm_fp8_block128_bf16(x, weight)
```

All public APIs have fake/meta registration and can be called inside a
`torch.compile(fullgraph=True)` region after loading the built artifact.

## Contract

- CUDA BF16 contiguous inputs.
- `dim` must be a positive multiple of 128.
- FP8 output dtype is `torch.float8_e4m3fn`.
- Scale dtype is FP32, with one value per row and 128 input channels.
- Raw widths that are not divisible by 128, such as 4304, are rejected. Pad to
  the model's blockwise GEMM width, such as 4352, and slice the downstream
  logical output where required.

The fused producers intentionally do not materialize an intermediate BF16
tensor before quantization. Correctness is therefore evaluated against the
mathematical BF16/FP32 reference with FP8 quantization metrics, while migration
parity is checked bitwise against the original FlashRT native CUDA entry.

## Validation

The source correctness matrix covers rows from 1 through 512 and transformer
widths from 1024 through 12288. It includes GROOT, Qwen3-VL, LingBot, Cosmos
and video-transformer shapes.

See `benchmarks/RESULTS.md`.
