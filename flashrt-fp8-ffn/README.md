# flashrt-fp8-ffn

Tensor-facing FlashRT FP8 GEMM and FFN blocks for Hugging Face `kernels`.

This package exports reusable FP8 FFN surfaces from the FlashRT serving stack
without exposing raw pointer APIs:

- `fp8_gemm_bf16`: per-tensor FP8 E4M3 GEMM with BF16 output.
- `fp8_linear_bias_gelu_quant_bf16`: FP8 GEMM, BF16 bias/GELU, and FP8 output
  quantization.
- `fp8_gelu_mlp_bf16`: complete GELU MLP block:
  `FP8 up GEMM -> bias/GELU -> FP8 quant -> FP8 down GEMM -> bias`.
- `bf16_fp8_gelu_mlp_bf16`: BF16 region entry that performs the static input
  quantization and the complete FP8 GELU MLP behind one traceable custom-op
  boundary.

The API is generic. PI0.5/GROOT/Wan-shaped demos live outside the package.

## Scope

Inputs and weights are row-major FP8 E4M3 tensors:

- activation/input: `(M, K)`
- weight: `(N, K)`
- output: `(M, N)` BF16
- scale tensors: device `float32` scalars

CUDA artifacts use `torch.float8_e4m3fn`. ROCm artifacts currently target AMD
CDNA3 `gfx942` and use `torch.float8_e4m3fnuz`. CDNA4/OCP-FP8 and RDNA targets
are not claimed by this package version.

The BF16-entry API in this revision is published in the current x86_64 CUDA
and ROCm build matrix. An older aarch64 directory retained by the incremental
Hub repository predates this API and is not a BF16-entry compatibility claim.
Build and validate a matching aarch64 variant before enabling this API there.

The first version uses per-tensor scales because this mirrors the production
FlashRT FP8 descale paths used by the current PI0.5/GROOT RTX frontends. Native
block-scaled FP8 and shape-locked megakernels should be separate follow-up
packages once their Tensor API and hardware scope are finalized.

## Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/flashrt-fp8-ffn", version=1)

x = torch.randn((512, 1024), device="cuda", dtype=torch.bfloat16)
w_up = torch.randn((4096, 1024), device="cuda", dtype=torch.bfloat16)
w_dn = torch.randn((1024, 4096), device="cuda", dtype=torch.bfloat16)

x_scale = torch.tensor([0.05], device="cuda")
up_scale = torch.tensor([0.04], device="cuda")
hidden_scale = torch.tensor([0.25], device="cuda")
dn_scale = torch.tensor([0.04], device="cuda")

w_up_fp8 = torch.clamp(w_up.float() / up_scale, -448, 448).to(torch.float8_e4m3fn)
w_dn_fp8 = torch.clamp(w_dn.float() / dn_scale, -448, 448).to(torch.float8_e4m3fn)

y = ops.bf16_fp8_gelu_mlp_bf16(
    x,
    w_up_fp8,
    torch.zeros((4096,), device="cuda", dtype=torch.bfloat16),
    w_dn_fp8,
    torch.zeros((1024,), device="cuda", dtype=torch.bfloat16),
    x_scale,
    up_scale,
    hidden_scale,
    dn_scale,
)
```

The BF16 entry accepts optional `input_fp8`, `hidden_bf16`, `hidden_fp8`, and
`out` buffers. Preallocate them for allocation-free CUDA Graph replay. Use a
static `pad_to` when a deployment intentionally pads M; the returned tensor is
sliced back to the logical row count. This is one Python/custom-op boundary,
not a claim that the two GEMMs and quantization execute as one CUDA launch.
Its input quantizer follows the FlashRT production arithmetic contract exactly:
`clamp(input.float() * (1.0 / input_scale), -fp8_max, fp8_max)`.

For dimensions divisible by four, the BF16 input producer, fused
bias/GELU/FP8 producer, and final BF16 bias use vectorized loads/stores. The
same APIs retain scalar fallbacks for other dimensions; callers do not need to
select an implementation.

The package registers a fake implementation for `torch.compile`. A static
region with preallocated scratch passes `torch.compile(fullgraph=True)` and
explicit CUDA Graph replay in the package correctness gate.

## Validation

Run from the repository root:

```bash
python flashrt-fp8-ffn/tests/test_fp8_ffn.py --backend source
python flashrt-fp8-ffn/benchmarks/benchmark.py --backend source --shapes headline --compile-baseline
python flashrt-fp8-ffn/benchmarks/benchmark.py --backend source --shapes all
python flashrt-fp8-ffn/benchmarks/benchmark_bf16_entry.py \
  --backend source --shapes all --compile-baseline
```

The package benchmark is reported against the PyTorch eager FP8 reference and a
segmented compile-stable `torch.compile` reference. The compiled reference is
timed only after its output is verified against eager output. The compile-stable
path graph-breaks the numerically sensitive `GELU -> FP8 requant` and final
BF16 bias/cast boundaries, while keeping the FP8 dequant GEMM regions compiled.
Model-block demos should be reported separately from full model generation
throughput claims.

`benchmark_bf16_entry.py` uses the fair region-boundary comparison: BF16
PyTorch eager, a verified full-graph BF16 `torch.compile` reference, the old
separate Python quantization path, FP8 kernel-only timing, and the new BF16
entry. It does not time FP8 dequantization inside the BF16 baseline. Migration
correctness requires exact equality with the established FlashRT staged ops;
the independent BF16 reference is reported separately as a quantization-quality
metric.
