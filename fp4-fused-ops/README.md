# fp4-fused-ops

FlashRT fused FP16-to-NVFP4 producer kernels for keeping low-bit transformer
and diffuser paths continuous.

The source kernels use the CUTLASS SM100-family block-scale layout shared by
SM110 and SM120. Published artifacts are architecture-specific. Native Thor
artifacts cover Torch 2.11 and 2.13 with CUDA 13.0 and are validated on Jetson
AGX Thor before upload.

These kernels turn FP16 residual/norm/gated activations directly into NVFP4
packed tensors plus CUTLASS-compatible SFA scale-factor buffers. They are meant
to feed adjacent FP4 GEMM kernels without returning to PyTorch elementwise
chains.

## Available Functions

- `sfa_size_bytes(rows, dim, is_sfb=False)`
- `rms_norm_fp4_sfa_fp16(x, packed=None, sfa=None)`
- `residual_add_rms_norm_fp4_sfa_fp16(residual, x, packed=None, sfa=None)`
- `residual_add_rms_norm_fp4_sfa_v2_fp16(residual, x, packed=None, sfa=None)`
- `residual_add_rms_norm_mul_fp4_sfa_fp16(residual, x, inv_s, packed=None, sfa=None)`
- `silu_mul_fp4_sfa_fp16(merged, packed=None, sfa=None)`
- `silu_mul_fp4_sfa_v2_fp16(merged, packed=None, sfa=None)`
- `silu_mul_mul_fp4_sfa_v2_fp16(merged, inv_s, packed=None, sfa=None)`
- `silu_mul_two_fp4_to_fp4(gate_packed, gate_sfa, up_packed, up_sfa, out_packed=None, out_sfa=None)`
- `silu_mul_two_mul_fp4_to_fp4(gate_packed, gate_sfa, up_packed, up_sfa, inv_s, out_packed=None, out_sfa=None)`
- `adaptive_rms_norm_nvfp4_fp16(x, style, packed=None, sfa=None, gate=None)`
- `gated_residual_adaptive_rms_norm_nvfp4_fp16(x, previous_gate, residual, style, ...)`
- `adaptive_rms_norm_fp8_static_fp16(x, style, scale, out=None, gate=None)`
- `gate_res_ada_rms_norm_quant_fp8_static_fp16(x, previous_gate, residual, style, scale, ...)`
- `adaptive_rms_norm_e0m3_fp16(x, style, use_rht=False, ...)`
- `gated_residual_adaptive_rms_norm_e0m3_fp16(x, previous_gate, residual, style, use_rht=False, ...)`
- `gelu_mul_e0m3_fp16(merged, use_rht=False, ...)`
- `residual_add_rms_norm_quant_nvfp4_swizzled_bf16(residual, x, weight, eps=1e-6, ...)`
- `relu2_quant_nvfp4_swizzled_fp16(x, ...)`
- `layer_norm_fp8_fp16(x, gamma, beta, eps=1e-5, out=None)`
- `layer_norm_nvfp4_fp16(x, gamma, beta, inv_s=None, eps=1e-5, ...)`
- `gelu_mul_nvfp4_fp16(merged, packed=None, sfa=None)`
- `dequantize_fp4_sfa_fp16(packed, sfa, out=None)`
- `quantize_bf16_to_nvfp4_linear(input, packed=None, scale_factors=None)`
- `rms_silu_nvfp4_ndhwc_bf16(x, gamma, awq_inv_scale=None, eps=1e-6, packed=None, scale_factors=None)`
- `bf16_rms_norm_ncdhw(x, gamma, bias=None, eps=1e-6, out=None)`
- `bf16_rms_silu_ncdhw(x, gamma, prev_cache=None, eps=1e-6, out=None, next_cache=None)`

Tensor contract:

- FP16 producer inputs are contiguous CUDA tensors.
- Packed FP4 tensors are `torch.uint8` with shape `(rows, dim / 2)`.
- SFA tensors are one-dimensional `torch.uint8` buffers sized by
  `sfa_size_bytes(rows, dim, False, device=...)`.
- `residual_add_*` updates `residual` in place.
- `silu_mul_*` expects `merged` shape `(rows, 2 * hidden)`.
- v1 shared-memory RMS producers support `dim <= 2048`; larger RMS producer
  shapes should use `residual_add_rms_norm_fp4_sfa_v2_fp16`.
- All dimensions must be divisible by 16. Unsupported shapes raise instead of
  silently taking a slow or partial path.
- PI0.5 adaptive RMSNorm producers currently require `dim=1024`; `style` is
  `(rows, 3 * dim)` containing scale, shift, and gate.
- Automatically allocated SFA buffers are zero-initialized because CUTLASS
  tile padding is intentionally not written by every producer.
- Linear NVFP4 uses E2M1 values and linear UE4M3 scale bytes per 16 channels.
- E0M3 APIs use the uniform signed-int4 codebook with optional per-16
  orthonormal Hadamard rotation; their packed/SFA layout is GEMM-compatible
  but their values must not be decoded as E2M1.
- NCDHW RMS kernels accept BF16 `(B,C,T,H,W)`, even `C <= 1024`; the fused
  NVFP4 producer requires `C % 128 == 0`.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/fp4-fused-ops", version=1, trust_remote_code=True)

merged = torch.randn((16, 4096), device="cuda", dtype=torch.float16)
packed, sfa = ops.silu_mul_fp4_sfa_v2_fp16(merged)

# Optional debug/validation path. The hot path should pass packed/SFA directly
# to the adjacent FP4 GEMM instead of dequantizing.
dequant = ops.dequantize_fp4_sfa_fp16(packed, sfa)

video = torch.randn((1, 128, 5, 9, 11), device="cuda", dtype=torch.bfloat16)
gamma = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
packed_video, linear_scales = ops.rms_silu_nvfp4_ndhwc_bf16(video, gamma)
```

## Validation

```bash
python fp4-fused-ops/tests/test_fp4_fused_ops.py --backend source --mode full
python fp4-fused-ops/benchmarks/benchmark.py --mode headline

# Thor production-model rows
python fp4-fused-ops/tests/test_fp4_fused_ops.py --backend installed \
  --mode thor-models \
  --artifact fp4-fused-ops/build/torch211-cxx11-cu130-aarch64-linux
python fp4-fused-ops/benchmarks/benchmark.py --backend installed \
  --mode thor-models \
  --artifact fp4-fused-ops/build/torch211-cxx11-cu130-aarch64-linux
```

Validation checks:

- in-place residual updates match the FP16 math contract exactly;
- FP4/SFA outputs dequantize within the documented NVFP4 quantization error
  envelope;
- v2 fast paths are checked against v1 where the v1 kernel is valid;
- unsupported shapes are rejected explicitly.
- new NCDHW/cache outputs and linear NVFP4 bytes match raw launchers bitwise.
