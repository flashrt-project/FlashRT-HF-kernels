# fp8-gemm

FlashRT native CUDA FP8 GEMV/GEMM kernels for low-latency transformer and
diffuser linear layers on Blackwell-class GPUs.

This package exposes the hand-tuned FP8 E4M3 decode and small-M kernels as
Tensor APIs for Hugging Face Kernel Hub. It is intended for model runtimes that
already hold activations and weights in FP8 and want a low-overhead BF16 output
linear path.

## Available Functions

- `fp8_linear_bf16(input, weight, alpha=1.0, out=None, variant=0)`
- `fp8_linear_residual_bf16(input, weight, residual, alpha=1.0, variant=0)`
- `select_fp8_linear_tile(m, n, k, variant=0)`

Tensor contract:

- `input`: `torch.float8_e4m3fn`, shape `(M, K)`, contiguous CUDA tensor.
- `weight`: `torch.float8_e4m3fn`, shape `(N, K)`, contiguous CUDA tensor.
- `out`: `torch.bfloat16`, shape `(M, N)`.
- `residual`: `torch.bfloat16`, shape `(1, N)` or `(N,)`, only supported for
  the `M=1` decode GEMV path.
- `K % 32 == 0`.
- `M == 1` uses dedicated GEMV. `2 <= M <= 64` uses small-M GEMM tiles.
- Build target is `sm_120a`; these kernels use Blackwell FP8 MMA instructions
  that are not valid for plain `sm_120` compilation.
- `alpha` is a host float. For per-tensor FP8 quantization, pass
  `float(input_scale * weight_scale)` from your static calibration metadata.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/fp8-gemm", version=1, trust_remote_code=True)

x = torch.randn((16, 4096), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
w = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)

y = ops.fp8_linear_bf16(x, w, alpha=1.0)
```

Decode residual path:

```python
x = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
w = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
residual = torch.zeros((1, 4096), device="cuda", dtype=torch.bfloat16)

ops.fp8_linear_residual_bf16(x, w, residual, alpha=1.0)
```

## Validation

```bash
python fp8-gemm/tests/test_fp8_gemm.py --backend source --mode full
python fp8-gemm/benchmarks/benchmark.py --backend source --mode headline
```

Public benchmark tables are only updated after source correctness, installed
artifact correctness, and shape/tile sweeps pass.
