# fp4-gemm

FlashRT native Blackwell NVFP4 A4W4 GEMM kernels.

This package consumes packed FP4 E2M1 tensors plus CUTLASS Sm1xx SFA/SFB scale
buffers and produces BF16 output. It is designed to pair with
`flashrt/fp4-fused-ops` and other static low-bit transformer/diffuser runtime
paths.

## Available Functions

- `sfa_size_bytes(rows, dim)`
- `quantize_fp4_sfa_fp16(x, packed=None, sfa=None, is_sfb=False)`
- `quantize_fp4_sfa_bf16(x, packed=None, sfa=None, is_sfb=False)`
- `quantize_e0m3_sfa_fp16(x, packed=None, sfa=None, is_sfb=False)`
- `dequantize_fp4_sfa_fp16(packed, sfa, out=None, is_sfb=False)`
- `nvfp4_gemm_bf16(a_packed, b_packed, sfa, sfb, alpha=1.0, out=None, variant=-1)`
- `nvfp4_gemm_fp16(a_packed, b_packed, sfa, sfb, alpha=1.0, out=None, variant=-1)`
- `nvfp4_gemm_geglu_nvfp4_fp16(a_packed, b_interleaved_packed, sfa, sfb, skinny=False, ...)`
- `nvfp4_gemm_bias_gelu_nvfp4_fp16(a_packed, b_packed, sfa, sfb, bias, ...)`
- `nvfp4_gemm_bias_residual_fp16(a_packed, b_packed, sfa, sfb, bias, residual, out=None)`
- `nvfp4_gemm_bias_bf16(a_packed, b_packed, sfa, sfb, bias, out=None)`
- `nvfp4_gemm_bias_residual_bf16(a_packed, b_packed, sfa, sfb, bias, residual, out=None)`
- `nvfp4_gemm_residual_bf16(a_packed, b_packed, sfa, sfb, residual, alpha=1.0, out=None)`
- `nvfp4_gemm_bias_gelu_bf16(a_packed, b_packed, sfa, sfb, bias, alpha=1.0, out=None)`
- `nvfp4_gemm_bias_gelu_nvfp4(a_packed, b_packed, sfa, sfb, bias, alpha=1.0, out_packed=None, out_sfa=None)`
- `nvfp4_gemm_streamk_bf16(a_packed, b_packed, sfa, sfb, alpha=1.0, out=None)`
- `nvfp4_gemm_streamk_bias_bf16(a_packed, b_packed, sfa, sfb, bias, alpha=1.0, out=None)`
- `fp4_w4a16_linear_bf16(...)` is retained as a compatibility alias
- `e0m3_weight_gemm_fp16(a_packed, b_packed, sfa, sfb, alpha=1.0, a_format=1, out=None)`
- `nvfp4_gemm_relu2_nvfp4(a_packed, b_packed, sfa, sfb, out_packed=None, out_sfa=None)`

## Tensor Contract

- `a_packed`: `torch.uint8`, shape `(M, K / 2)`.
- `b_packed`: `torch.uint8`, shape `(N, K / 2)`.
- `sfa`: `torch.uint8`, CUTLASS SFA layout for `(M, K)`.
- `sfb`: `torch.uint8`, CUTLASS SFB layout for `(N, K)`.
- output: `torch.bfloat16`, shape `(M, N)`.
- Native SM110 PI0.5 epilogues use FP16 outputs/producers; their dtype is
  explicit in the function name.
- `K` must be divisible by 16.
- Targets: Blackwell `sm_110a` (Jetson AGX Thor, CUDA 13+) and `sm_120a`
  (RTX Blackwell, CUDA 12.8+).

`variant` selects the CUTLASS schedule:

- `-1`: architecture-aware auto-dispatch (public default).
- `0`: default `<128,128,256>` cooperative schedule.
- `1`: widen `<128,256,128>` schedule, intended for very large `N`.
- `2`: pingpong schedule for A/B testing shape-specific wins.

The canonical linear API and FP4/SFA quantize/dequantize helpers are available
on both SM110 and SM120. SM110 additionally provides the GROOT N1.7 production
epilogues `nvfp4_gemm_bias_bf16`, `nvfp4_gemm_bias_residual_bf16`, and
`nvfp4_gemm_bias_gelu_nvfp4`. The latter emits packed FP4 plus CUTLASS SFA so
the following projection can consume it without a BF16 materialization and a
standalone quantization launch. Stream-K and the older BF16 GELU epilogue keep
their existing SM120 dispatch and reject unsupported architectures explicitly.

The SM110 release gate includes the production `(M,N,K)` shapes
`(41,4608,1536)`, `(41,6144,1536)`, and `(41,1536,6144)`, plus the legacy
`M=51` compatibility row. The kernels are the native sources used by FlashRT's
GROOT N1.7 Thor NVFP4 pipeline.

PI0.5 Thor coverage additionally includes FP16 linear projections, compact
GeGLU-to-NVFP4, and SigLIP bias+GELU / bias+residual epilogues. SigLIP's
logical hidden width `4304` is physically zero-padded to `4320` for the FP4
TMA contract; direct `4304` GEMM calls are unsupported.
E0M3 weight GEMM and the ReLU-squared FP4-output epilogue are SM110-only.
`a_format=1` consumes E2M1 activations; `a_format=0` consumes E0M3 activations.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/fp4-gemm", version=1, trust_remote_code=True)

x = torch.randn((32, 256), device="cuda", dtype=torch.float16)
w = torch.randn((512, 256), device="cuda", dtype=torch.float16)

a_packed, sfa = ops.quantize_fp4_sfa_fp16(x, is_sfb=False)
b_packed, sfb = ops.quantize_fp4_sfa_fp16(w, is_sfb=True)

y = ops.nvfp4_gemm_bf16(a_packed, b_packed, sfa, sfb, alpha=1.0)
```

For BF16 model activations, use the direct producer so the hot path does not
materialize an intermediate FP16 tensor:

```python
x_bf16 = torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16)
a_packed, sfa = ops.quantize_fp4_sfa_bf16(x_bf16)
```

The BF16 entry writes the same E2M1 bytes and CUTLASS SFA/SFB layout as
`quantize_fp4_sfa_fp16(x_bf16.to(torch.float16))` for finite FP16-range
inputs. It is an additive API; the existing FP16 producer remains unchanged.

The quantize/dequantize helpers are included for examples and validation. A
production runtime should keep weights prepacked and should avoid quantizing in
the hot path unless that producer kernel is part of the intended low-bit block.

Use the bias/GELU and residual variants to avoid returning to BF16
elementwise code between low-bit GEMMs. Stream-K variants are selected only
for the validated large down-projection shapes; unsupported shapes reject
rather than silently selecting a losing schedule.

## Validation

```bash
python fp4-gemm/tests/test_fp4_gemm.py --backend source --mode full
python fp4-gemm/tests/test_fp4_gemm.py --backend installed --mode full \
  --artifact fp4-gemm/build/torch211-cxx11-cu128-x86_64-linux
python fp4-gemm/benchmarks/benchmark.py --backend installed --mode headline \
  --artifact fp4-gemm/build/torch211-cxx11-cu128-x86_64-linux

# Thor model-shape gate
python fp4-gemm/tests/test_fp4_gemm.py --backend installed \
  --mode thor-models \
  --artifact fp4-gemm/build/torch211-cxx11-cu130-aarch64-linux
```

The correctness reference dequantizes the same FP4/SFA and FP4/SFB inputs used
by the kernel, then computes the PyTorch GEMM reference from those dequantized
low-bit values.

The producer gate also checks the BF16 direct entry byte-for-byte against the
established FP16 compatibility chain at decode widths 5120, 6144 and 17408,
plus multi-row activation and SFB layouts.

Release artifacts cover the standard x86 matrix (Torch 2.11/2.12/2.13 over
the supported CUDA 12.8/13.x variants) and native Thor aarch64 builds for
Torch 2.11 and 2.13 with CUDA 13.0. Consumers should request `version=1`.
