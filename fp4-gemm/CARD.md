# flashrt/fp4-gemm

FlashRT native Blackwell NVFP4 GEMM kernels. The package includes packed A4W4
paths and an explicitly named small-M W4A16 path that keeps activations BF16.

## Functions

- `sfa_size_bytes`
- `capabilities`
- `quantize_fp4_sfa_fp16`
- `quantize_fp4_sfa_bf16`
- `quantize_fp4_sfa_mse_fp16`
- `quantize_fp4_sfa_mse_bf16`
- `quantize_fp4_sfa_padded_bf16`
- `pack_nvfp4_weight_bf16`
- `quantize_e0m3_sfa_fp16`
- `dequantize_fp4_sfa_fp16`
- `nvfp4_gemm_bf16`
- `nvfp4_gemm_fp16`
- `nvfp4_gemm_variant_bf16`
- `nvfp4_gemm_nvfp4`
- `nvfp4_gemm_geglu_nvfp4_fp16`
- `cutlass_fp4_gemm_geglu_il_hw_v10`
- `nvfp4_gemm_bias_gelu_nvfp4_fp16`
- `nvfp4_gemm_bias_residual_fp16`
- `nvfp4_gemm_bias_bf16`
- `nvfp4_gemm_bias_residual_bf16`
- `nvfp4_gemm_residual_bf16`
- `nvfp4_gemm_bias_gelu_bf16`
- `nvfp4_gemm_bias_gelu_nvfp4`
- `nvfp4_gemm_streamk_bf16`
- `nvfp4_gemm_streamk_bias_bf16`
- `fp4_w4a16_linear_bf16` (compatibility alias)
- `adopt_nvfp4_w4a16_marlin`
- `allocate_w4a16_marlin_workspace`
- `nvfp4_w4a16_marlin_bf16`
- `fp4_repack_b_interleaved_sm120`
- `fp4_w4a4_gemv_warpsplit_interleaved_bf16`
- `fp4_w4a4_gemm_warpsplit_mrows_bf16`
- `nvfp4_gemm_m256_workspace_size`
- `nvfp4_gemm_m256_bf16`
- `e0m3_weight_gemm_fp16`
- `nvfp4_gemm_relu2_nvfp4`

## Example

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/fp4-gemm", version=1, trust_remote_code=True)

x = torch.randn((32, 256), device="cuda", dtype=torch.float16)
w = torch.randn((512, 256), device="cuda", dtype=torch.float16)

a, sfa = ops.quantize_fp4_sfa_fp16(x, is_sfb=False)
b, sfb = ops.quantize_fp4_sfa_fp16(w, is_sfb=True)
y = ops.nvfp4_gemm_bf16(a, b, sfa, sfb)
```

For M=1 decode, repack each static weight once and retain the interleaved
buffer beside the original packed weight:

```python
b_interleaved = ops.fp4_repack_b_interleaved_sm120(b)
y = ops.fp4_w4a4_gemv_warpsplit_interleaved_bf16(
    a, b_interleaved, sfa, sfb, warps=8, stages=3
)
```

Speculative verify batches with `1 <= M <= 16` can reuse the standard packed
weight without an interleaved duplicate:

```python
y = ops.fp4_w4a4_gemm_warpsplit_mrows_bf16(
    a, b, sfa, sfb, warps=2, stages=6
)
```

For accuracy-sensitive logits or verification projections, adopt standard
ModelOpt NVFP4 weights once and call the W4A16 tier with BF16 activations:

```python
w, s, g, workspace = ops.adopt_nvfp4_w4a16_marlin(
    weight_packed, weight_scale, weight_scale_2
)
y = ops.nvfp4_w4a16_marlin_bf16(
    x_bf16, w, s, g, workspace=workspace, out=out_bf16
)
```

The large-M tier uses a caller-owned workspace. Allocate it before CUDA Graph
capture and pass the same tensor on every replay:

```python
workspace_bytes = ops.nvfp4_gemm_m256_workspace_size(a, b, sfa, sfb)
workspace = torch.empty(workspace_bytes, device=a.device, dtype=torch.uint8)
y = ops.nvfp4_gemm_m256_bf16(a, b, sfa, sfb, workspace=workspace)
```

The current persistent M256 schedule reports exactly zero workspace bytes.
The helper has a matching FakeTensor implementation, and the convenience
wrapper keeps this scalar query outside `torch.compile(fullgraph=True)`.

BF16 activations should use the direct producer to avoid a separate cast and
copy before every low-bit projection:

```python
x = torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16)
a, sfa = ops.quantize_fp4_sfa_bf16(x)
```

## Notes

- Blackwell `sm_110a` with CUDA 13+ and `sm_120a` with CUDA 12.8+.
- Inputs are packed FP4 E2M1 plus CUTLASS Sm1xx SFA/SFB scale buffers.
- Output is BF16.
- Read `capabilities()` instead of duplicating scale-factor layout or alignment
  constants in a runtime integration. Unsupported calls raise; output is never
  silently left undefined.
- `variant=-1` is the architecture-aware production auto-dispatch;
  `variant=0/1/2` expose diagnostic default, widen, and pingpong schedules.
- The canonical BF16-output GEMM, fused bias GEMM, and FP4 pack/unpack helpers
  support SM110 and SM120. SM110 also supports bias+residual and bias+GELU-to-FP4
  production epilogues used by the GROOT N1.7 Thor pipeline.
- Dimensions used by Blackwell NVFP4 TMA are physically aligned to 32. For
  logical widths such as SigLIP 4304, use the bind-time padding/packing helpers
  to create static 4320 tensors; the GEMM hot path performs no padding.
- `nvfp4_gemm_m256_bf16` requires M>=512. Production qualification currently
  covers `(N,K)=(17408,5120),(5120,17408),(12288,5120)` on RTX 5090. Read
  `capabilities()` before dispatch; `(16384,5120)` is diagnostic only.
- `fp4_w4a4_gemm_warpsplit_mrows_bf16` is SM120-only, accepts standard
  row-major packed E2M1 weights, and requires `1<=M<=16`, `N%8==0`, and
  `K%(64*warps)==0`. Unsupported calls raise before launch.
- `nvfp4_w4a16_marlin_bf16` is SM120-only and requires `1<=M<=16`,
  `K%128==0`, and `N%64==0`. Its output/workspace are caller-owned, so the
  runtime call is graph safe. The implementation is adapted from the
  Apache-2.0 vLLM/Marlin backend and has no runtime dependency on vLLM.
