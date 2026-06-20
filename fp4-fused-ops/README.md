# fp4-fused-ops

FlashRT fused FP16-to-NVFP4 producer kernels for keeping low-bit transformer
and diffuser paths continuous.

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
- `dequantize_fp4_sfa_fp16(packed, sfa, out=None)`

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
```

## Validation

```bash
python fp4-fused-ops/tests/test_fp4_fused_ops.py --backend source --mode full
python fp4-fused-ops/benchmarks/benchmark.py --mode headline
```

Validation checks:

- in-place residual updates match the FP16 math contract exactly;
- FP4/SFA outputs dequantize within the documented NVFP4 quantization error
  envelope;
- v2 fast paths are checked against v1 where the v1 kernel is valid;
- unsupported shapes are rejected explicitly.
