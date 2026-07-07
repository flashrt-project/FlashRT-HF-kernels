# int8-transformer-primitives

Generic INT8 primitives for transformer and diffusion-model blocks.

Hub repo: `flashrt/int8-transformer-primitives`

## Public API

- `quantize_int8_static_bf16(input, scale) -> int8`
- `quantize_int8_rowwise_bf16(input) -> (int8, scales)`
- `quantize_int8_rowwise_static_bf16(input, scales) -> int8`
- `rms_norm_quantize_int8_rowwise_bf16(x, weight, eps=1e-6) -> (int8, scales)`
- `residual_add_rms_norm_quantize_int8_rowwise_bf16(residual, x, weight, eps=1e-6) -> (int8, scales)`
- `int8_rowwise_linear_bf16(input_i8, weight_i8, input_scale, weight_scale, variant=0) -> bf16`
- `int8_silu_gated_linear_bf16(input_i8, up_weight_i8, input_scale, weight_scale, gate) -> bf16`

The package is intentionally model-neutral. It exposes Tensor APIs suitable for
Transformers/Diffusers integration paths that want explicit INT8 producer and
consumer kernels without depending on FlashRT runtime internals.

## Example

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/int8-transformer-primitives", version=1)

x = torch.randn((64, 2048), device="cuda", dtype=torch.bfloat16)
w = torch.randn((4096, 2048), device="cuda", dtype=torch.bfloat16)
w_i8, w_scale = ops.quantize_int8_rowwise_bf16(w)
x_i8, x_scale = ops.quantize_int8_rowwise_bf16(x)
y = ops.int8_rowwise_linear_bf16(x_i8, w_i8, x_scale, w_scale)
```

## Shape contract

- Activations use row-major `(M, K)` INT8 plus per-row FP32 scales `(M,)`.
- Weights use row-major `(N, K)` INT8 plus per-output-channel FP32 scales `(N,)`.
- `K` must be divisible by 16 and `N` must be divisible by 8 for CUTLASS INT8 GEMM.
- Outputs are BF16 `(M, N)`.

## Validation

Correctness is tested against PyTorch BF16/FP32 reference formulas. INT8
quantization uses round-to-nearest-even and clamps to `[-127, 127]`.

See `benchmarks/RESULTS.md` for current local RTX 5090 source benchmark data.
