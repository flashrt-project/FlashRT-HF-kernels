# flashrt-fp8-swiglu-ffn

Tensor-facing FlashRT FP8 GeGLU/SwiGLU FFN kernels for Hugging Face `kernels`.

This package is the second-batch VLA/runtime package for Gemma-style FFN
islands:

```text
FP8 input -> FP8 gate/up GEMM -> GeGLU/SwiGLU gate activation -> FP8 requant -> FP8 down GEMM -> BF16 output
```

The APIs are generic Tensor APIs. Model-specific PI0.5/GROOT/Wan runtime wiring
lives in `demos/`, not in the package entry points.

## Exported APIs

- `fp8_gemm_bf16(input, weight, input_scale, weight_scale, out=None)`
- `silu_mul_merged_quantize_fp8_static_bf16(gate_up_bf16, output_scale, out_fp8=None)`
- `gelu_mul_merged_quantize_fp8_static_bf16(gate_up_bf16, output_scale, out_fp8=None)`
- `fp8_swiglu_mlp_bf16(input, gate_up_weight, down_weight, input_scale, gate_up_weight_scale, hidden_scale, down_weight_scale, gate_up_bf16=None, hidden_fp8=None, out=None)`
- `fp8_geglu_mlp_bf16(input, gate_up_weight, down_weight, input_scale, gate_up_weight_scale, hidden_scale, down_weight_scale, gate_up_bf16=None, hidden_fp8=None, out=None)`

Tensor conventions:

- `input`: FP8 E4M3, shape `(M, K)`
- `gate_up_weight`: FP8 E4M3, shape `(2 * H, K)`, row layout `[gate_weight | up_weight]`
- `down_weight`: FP8 E4M3, shape `(N, H)`
- scale tensors: CUDA `float32` scalar tensors
- output: BF16, shape `(M, N)`

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/flashrt-fp8-swiglu-ffn", version=1)

x_scale = torch.tensor([0.05], device="cuda")
gate_up_scale = torch.tensor([0.04], device="cuda")
hidden_scale = torch.tensor([0.25], device="cuda")
down_scale = torch.tensor([0.04], device="cuda")

x = torch.randn((10, 1024), device="cuda", dtype=torch.bfloat16)
gate_up_w = torch.randn((8192, 1024), device="cuda", dtype=torch.bfloat16)
down_w = torch.randn((1024, 4096), device="cuda", dtype=torch.bfloat16)

x_fp8 = torch.clamp(x.float() / x_scale, -448, 448).to(torch.float8_e4m3fn)
gate_up_w_fp8 = torch.clamp(gate_up_w.float() / gate_up_scale, -448, 448).to(torch.float8_e4m3fn)
down_w_fp8 = torch.clamp(down_w.float() / down_scale, -448, 448).to(torch.float8_e4m3fn)

y = ops.fp8_geglu_mlp_bf16(
    x_fp8,
    gate_up_w_fp8,
    down_w_fp8,
    x_scale,
    gate_up_scale,
    hidden_scale,
    down_scale,
)
```

Use `fp8_geglu_mlp_bf16` for Gemma/PI0.5-style
`gelu_pytorch_tanh(gate) * up`. Use `fp8_swiglu_mlp_bf16` for true
`SiLU(gate) * up` blocks.

For hot-path runtime use, load the kernel once, keep weights/scales resident,
preallocate `gate_up_bf16`, `hidden_fp8`, and `out`, and capture the repeated
block with CUDA Graph if the shape is static.

## Validation

Source-extension smoke:

```bash
python flashrt-fp8-swiglu-ffn/tests/test_fp8_swiglu_ffn.py --backend source --mode smoke
```

Full source-extension correctness:

```bash
python flashrt-fp8-swiglu-ffn/tests/test_fp8_swiglu_ffn.py --backend source --mode full
```

Benchmark:

```bash
python flashrt-fp8-swiglu-ffn/benchmarks/benchmark.py --backend source --shapes headline
python flashrt-fp8-swiglu-ffn/benchmarks/benchmark.py --backend source --shapes all
```

The correctness gate reports `max_abs`, `mean_abs`, `p99_abs`, cosine
similarity, and relative error. Benchmarks compare against PyTorch eager and an
optional compile-stable PyTorch reference only after the reference is verified.
