# fp4-gemm

FlashRT native Blackwell NVFP4 A4W4 GEMM kernels.

This package consumes packed FP4 E2M1 tensors plus CUTLASS Sm1xx SFA/SFB scale
buffers and produces BF16 output. It is designed to pair with
`flashrt/fp4-fused-ops` and other static low-bit transformer/diffuser runtime
paths.

## Available Functions

- `sfa_size_bytes(rows, dim)`
- `quantize_fp4_sfa_fp16(x, packed=None, sfa=None, is_sfb=False)`
- `dequantize_fp4_sfa_fp16(packed, sfa, out=None, is_sfb=False)`
- `nvfp4_gemm_bf16(a_packed, b_packed, sfa, sfb, alpha=1.0, out=None, variant=-1)`
- `nvfp4_gemm_residual_bf16(a_packed, b_packed, sfa, sfb, residual, alpha=1.0, out=None)`
- `nvfp4_gemm_bias_gelu_bf16(a_packed, b_packed, sfa, sfb, bias, alpha=1.0, out=None)`
- `nvfp4_gemm_bias_gelu_nvfp4(a_packed, b_packed, sfa, sfb, bias, alpha=1.0, out_packed=None, out_sfa=None)`
- `nvfp4_gemm_streamk_bf16(a_packed, b_packed, sfa, sfb, alpha=1.0, out=None)`
- `nvfp4_gemm_streamk_bias_bf16(a_packed, b_packed, sfa, sfb, bias, alpha=1.0, out=None)`
- `fp4_w4a16_linear_bf16(...)` is retained as a compatibility alias

## Tensor Contract

- `a_packed`: `torch.uint8`, shape `(M, K / 2)`.
- `b_packed`: `torch.uint8`, shape `(N, K / 2)`.
- `sfa`: `torch.uint8`, CUTLASS SFA layout for `(M, K)`.
- `sfb`: `torch.uint8`, CUTLASS SFB layout for `(N, K)`.
- output: `torch.bfloat16`, shape `(M, N)`.
- `K` must be divisible by 16.
- Targets: Blackwell `sm_110a` (Jetson AGX Thor, CUDA 13+) and `sm_120a`
  (RTX Blackwell, CUDA 12.8+).

`variant` selects the CUTLASS schedule:

- `-1`: architecture-aware auto-dispatch (public default).
- `0`: default `<128,128,256>` cooperative schedule.
- `1`: widen `<128,256,128>` schedule, intended for very large `N`.
- `2`: pingpong schedule for A/B testing shape-specific wins.

The canonical linear API and FP4/SFA quantize/dequantize helpers are available
on both SM110 and SM120. The residual, bias/GELU, FP4-output and Stream-K
epilogues are currently SM120-only. SM110 runtimes compose the linear API with
the production producers in `flashrt/fp4-fused-ops`; requesting an SM120-only
epilogue on SM110 raises an explicit architecture error.

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
