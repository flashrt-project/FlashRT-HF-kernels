# padded-fp8-producers

Padding-aware fused producers for static-shape transformer and VLA regions.
The kernels fuse normalization or SwiGLU activation with static FP8
quantization and zero-fill the tail rows required by a downstream tiled GEMM.

## Load from Kernel Hub

```python
import torch
from kernels import get_kernel

k = get_kernel(
    "flashrt/padded-fp8-producers",
    version=1,
    trust_remote_code=True,
)

x = torch.randn(1, 51, 2048, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(2048, device="cuda", dtype=torch.bfloat16)
gamma = torch.zeros(1, 2048, device="cuda", dtype=torch.bfloat16)
beta = torch.zeros_like(gamma)
scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)

x_fp8 = k.adaptive_rms_norm_quant_fp8_padded_bf16(
    x, weight, gamma, beta, scale, padded_rows=64
)
assert x_fp8.shape == (1, 64, 2048)
```

## Public functions

- `adaptive_rms_norm_quant_fp8_padded_bf16`
- `residual_add_adaptive_rms_norm_quant_fp8_padded_bf16`
- `swiglu_quant_fp8_padded_bf16`
- `swiglu_merged_quant_fp8_padded_bf16`
- `swiglu_merged_quant_fp8_padded_fp16`

All scale tensors are one-element FP32 static activation scales. Inputs must be
contiguous CUDA tensors. FP8 outputs use `torch.float8_e4m3fn`. `padded_rows`
must be at least the logical row count; all added rows are exactly zero.

Pass preallocated `output` and `residual_out` tensors on CUDA Graph hot paths.
The convenience allocation path is intended for setup and eager use.

See [VALIDATION.md](VALIDATION.md) for the supported shape contract and
[benchmarks/RESULTS.md](benchmarks/RESULTS.md) for qualification data.
