# adaptive-layernorm-producers

FlashRT native CUDA producer kernels for DiT/Wan-style adaptive LayerNorm
blocks.

This package fuses the producer side of the block:

```text
LayerNorm / adaptive LayerNorm modulation -> low-precision activation
```

into a single Kernel Hub operator. It is intended for transformer and diffuser
runtimes where the following GEMM already consumes FP8 or NVFP4 activations.
Without this fusion, the runtime pays for separate normalization, modulation,
BF16 materialization, and quantization launches before the GEMM.

The source kernels are derived from the production FlashRT runtime:

- `official/FlashRT/csrc/quantize/ada_layer_norm_fp8.cu`
- `official/FlashRT/csrc/kernels/dit_bf16.cu::layer_norm_no_affine_fp8_static_bf16`

The full FlashRT model runtime and serving pipeline live upstream at
[LiangSu8899/FlashRT](https://github.com/LiangSu8899/FlashRT).

## Available Functions

- `ada_layer_norm_quant_fp8_bf16(x, scale, shift, act_scale, eps=1e-5, out=None)`
- `ada_layer_norm_quant_fp8_modfp8_bf16(x, scale_fp8, shift_fp8, scale_deq, shift_deq, act_scale, eps=1e-5, out=None)`
- `awq_ada_layer_norm_quant_fp8_bf16(x, scale, shift, inv_s, act_scale, eps=1e-5, out=None)`
- `ada_layer_norm_quant_nvfp4_swizzled_bf16(x, scale, shift, eps=1e-5, packed=None, sf_swizzled=None)`
- `ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16(x, scale_fp8, shift_fp8, scale_deq, shift_deq, eps=1e-5, packed=None, sf_swizzled=None)`
- `layer_norm_no_affine_quant_fp8_static_bf16(x, act_scale, eps=1e-5, out=None)`
- `swizzled_sf_size(rows, dim)`

## Tensor Contract

- `x`: contiguous CUDA `torch.bfloat16`, shape `(rows, dim)`.
- `scale`, `shift`, `inv_s`: contiguous CUDA `torch.bfloat16`, shape `(dim,)`.
- `scale_fp8`, `shift_fp8`: contiguous CUDA `torch.float8_e4m3fn`, shape `(dim,)`.
- `scale_deq`, `shift_deq`, `act_scale`: CUDA `torch.float32` scalar tensors.
- FP8 outputs use `torch.float8_e4m3fn`, shape `(rows, dim)`.
- NVFP4 packed output uses `torch.uint8`, shape `(rows, dim // 2)`.
- NVFP4 scale output uses the FlashRT/CUTLASS 128x4 swizzled uint8 scale
  layout. Allocate with `swizzled_sf_size(rows, dim)`.
- `dim` must be even for FP8 outputs and divisible by 16 for NVFP4 outputs.
- The package targets CUDA 12.8+ and Blackwell-class deployment paths.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/adaptive-layernorm-producers", version=1, trust_remote_code=True)

rows, dim = 2520, 3072
x = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
scale = torch.zeros((dim,), device="cuda", dtype=torch.bfloat16)
shift = torch.zeros((dim,), device="cuda", dtype=torch.bfloat16)
act_scale = torch.tensor([0.025], device="cuda", dtype=torch.float32)

x_fp8 = ops.ada_layer_norm_quant_fp8_bf16(x, scale, shift, act_scale)
```

Static-buffer usage for CUDA Graph capture:

```python
out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
ops.ada_layer_norm_quant_fp8_bf16(x, scale, shift, act_scale, out=out)
```

NVFP4 producer:

```python
packed, sf = ops.ada_layer_norm_quant_nvfp4_swizzled_bf16(x, scale, shift)
```

## Validation

```bash
python adaptive-layernorm-producers/tests/test_adaptive_layernorm_producers.py --backend source --mode full
python adaptive-layernorm-producers/benchmarks/benchmark.py --backend source --iters 100
```

The validation suite checks exact FP8 output for small producer shapes, exact
NVFP4 packed/scalefactor output for representative shapes, and strict FP8
boundary accounting for long video shapes. The long-shape policy requires
`p99_abs == 0` and only a tiny count of adjacent FP8-code boundary differences
against the eager reference.

See `VALIDATION.md` and `benchmarks/RESULTS.md` for the current local RTX 5090
source-build results.
