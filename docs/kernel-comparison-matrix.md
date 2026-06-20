# Kernel Comparison Matrix

This document defines the comparison matrix for every v1 FlashRT HF kernel
package. It is stricter than a single speedup table: each kernel must state
which baselines were measured, which baselines are still pending, and which
claims are allowed from the current evidence.

## Claim Levels

| Level | Meaning | Public use |
| --- | --- | --- |
| Correctness | Output agrees with a deterministic or PyTorch reference | Required before any timing table |
| Readable baseline | PyTorch eager tensor expression | Useful for API clarity and HF benchmark compatibility |
| Compiler baseline | `torch.compile` reference with compile time excluded | Required only when the compiled reference is verified equivalent to eager |
| Unfused strong path | Same operation as separate CUDA or FlashRT launches | Required for fusion-specific headline claims when available |
| Library baseline | cuBLASLt, CUTLASS, cuDNN, FlashAttention, FlashInfer, or another strong public implementation | Required for GEMM, attention, and low-bit headline claims |
| Model-block | HF-style model subgraph or demo path | Required for community/showcase claims |

Speedups against PyTorch eager alone are support evidence, not headline
evidence, unless the operation is purely a launch-bound tensor-op fusion and a
verified-equivalent `torch.compile` baseline is also reported. A compiled
reference that changes fake-quant or low-bit rounding behavior is not a valid
baseline.

## Result Labels

Every benchmark row should carry one of these labels:

| Label | Meaning |
| --- | --- |
| `headline` | Correctness passed and the strongest applicable baseline was measured |
| `support` | Correctness passed and the row is useful, but a stronger baseline is still pending |
| `compatibility` | Correctness coverage for a shape that is not performance-promoted |
| `diagnostic` | Internal tuning data, tile sweep, or suspected measurement artifact |
| `reject` | Correct but too slow for the advertised fast path |

Do not publish `diagnostic` or `reject` rows as package headline tables.

## Package Matrix

### `flashrt-gemm-epilogues`

Primary kernels:

- `bias_gelu_quantize_fp8_static_bf16`
- `gelu_quantize_fp8_static_bf16`
- `channel_scale_quantize_fp8_static_bf16`
- selected BF16 GEMM epilogue wrappers

Required comparisons:

| Scope | Baselines | Headline gate |
| --- | --- | --- |
| FP8 quant epilogue chains | PyTorch eager, `torch.compile`, optional unfused FlashRT/CUDA chain | `torch.compile` measured and correctness has FP8 byte or fake-quant parity |
| Channel-scale quantization | PyTorch eager, `torch.compile`, memory bandwidth estimate | `torch.compile` measured and bandwidth is reported |
| BF16 GEMM + bias/GELU | `torch.addmm`, compiled `torch.addmm`, cuBLASLt or CUTLASS fused/unfused path | Do not headline shapes that only beat eager |

Good public story:

- FP8 quant epilogues are the strongest v1 surface.
- BF16 GEMM epilogue wrappers remain shape-specific unless the cuBLASLt/CUTLASS
  comparison is strong.

### `flashrt-fp8-ffn`

Primary kernels:

- `fp8_gemm_bf16`
- `fp8_linear_bias_gelu_quant_bf16`
- `fp8_gelu_mlp_bf16`

Required comparisons:

| Scope | Baselines | Headline gate |
| --- | --- | --- |
| Full FP8 GELU MLP | PyTorch eager FP8 reference, segmented compile-stable `torch.compile` reference, cuBLASLt/CUTLASS or FlashRT internal strong path where applicable | Eager correctness passes; compiled-reference output verifies against eager before timing |
| FP8 GEMM sublayer | cuBLASLt/CUTLASS or existing FlashRT internal GEMM path | Strong library/internal path measured for headline GEMM claims |
| Model-block FFN shapes | PI0.5/GROOT/VLA shape families with layer counts | Shape family and precision gate reported separately from full-model throughput |

Good public story:

- Full FFN correctness and eager-reference speedups are valid current support
  evidence.
- The package uses a segmented compile-stable reference for `torch.compile`
  timing. It graph-breaks the `GELU -> FP8 requant` and final BF16 bias/cast
  boundaries because raw default-Inductor fusion is not bit-equivalent to eager
  at those quantization boundaries.

### `flashrt-vla-video`

Primary kernels:

- `q_norm_rope_bf16`
- `k_norm_rope_v_cache_bf16`
- `qkv_split_norm_rope_bf16`

Required comparisons:

| Scope | Baselines | Headline gate |
| --- | --- | --- |
| Q/K RMSNorm + RoPE | PyTorch eager, `torch.compile`, separate unfused launch chain if available | `torch.compile` measured on token/head grid |
| Packed QKV split + Q/K norm + RoPE | PyTorch eager, `torch.compile`, model-block postprocess path | `torch.compile` measured and Wan/VLA shape family shown |
| Attention subpath | Same SDPA/FlashAttention backend on both paths | Only the postprocess delta is claimed; attention backend is held constant |
| Model-block demo | HF-style callable demo for Wan/VLA/video shapes | End-to-end subpath speedup reported separately from microkernel speedup |

Good public story:

- This package targets a real community gap: VLA/video/diffusion postprocess
  fusion around QKV, normalization, RoPE, and cache staging.
- The strongest current evidence is the Wan/video demo because it compares both
  postprocess-only and self-attention sublayer paths.

### `flashrt-nvfp4`

Primary kernels:

- `nvfp4_sf_linear_to_swizzled`
- candidate NVFP4/FP4 GEMM epilogues

Required comparisons:

| Scope | Baselines | Headline gate |
| --- | --- | --- |
| Scale-factor layout helper | PyTorch/CUDA tensor reshape reference, byte-parity reference, optional `torch.compile` if tensor-only | Correctness byte parity plus latency/bandwidth |
| NVFP4/FP4 GEMM epilogue | CUTLASS/cuBLASLt FP4 path, unfused GEMM plus separate epilogue, PyTorch fake-quant reference | Strong library or unfused CUDA path measured |
| SM120 hardware scope | Same artifact on RTX 5090 first, then other Blackwell hardware | Keep CUDA 12.8+ SM120 label until broader source path exists |

Good public story:

- Layout helpers are compatibility and enablement evidence.
- GEMM/FP4 epilogues require strong-library comparison before headline use.

### `flashrt-smallm-gemm`

Primary kernels:

- `nvfp4_w4a4_decode_matvec_bf16out`
- candidate small-M warpsplit and tiny FP8 GEMM paths

Required comparisons:

| Scope | Baselines | Headline gate |
| --- | --- | --- |
| Decode `M=1` W4A4 matvec | PyTorch dequant+matmul, compiled dequant+matmul, CUTLASS/cuBLASLt or FlashRT internal low-bit baseline | Strong low-bit baseline measured |
| Small-M W4A4 GEMM | PyTorch readable baseline, strong low-bit library/internal baseline | Dispatch boundaries benchmarked |
| Tiny FP8 families | cuBLASLt FP8 or known strong FlashRT baseline where applicable | Fixed-shape families are faster than strong path |

Good public story:

- Do not present slow compatibility rows as performance rows.
- This package becomes public-headline material only when the selected
  decode/small-M shapes beat a credible low-bit baseline.

### `flashrt-fused-quant`

Primary kernels:

- `silu_mul_quant_nvfp4_swizzled_bf16`
- `silu_mul_merged_quant_nvfp4_swizzled_bf16`
- candidate RMSNorm/residual quantization kernels

Required comparisons:

| Scope | Baselines | Headline gate |
| --- | --- | --- |
| SwiGLU quantization | PyTorch eager, `torch.compile`, effective memory bandwidth | `torch.compile` measured and bandwidth reported |
| Split vs merged gate/up | Same shape grid, same output layout, separate rows in results | Faster variant selected per shape family |
| Residual/RMSNorm quantization | PyTorch readable chain, compiled chain, unfused CUDA chain if available | Aliasing correctness plus unfused-chain comparison |

Good public story:

- This is a memory-bandwidth and launch-count package.
- Report latency and GB/s together; speedup without bandwidth is not enough.

### `fp4-fused-ops`

Primary kernels:

- `rms_norm_fp4_sfa_fp16`
- `residual_add_rms_norm_fp4_sfa_fp16`
- `residual_add_rms_norm_fp4_sfa_v2_fp16`
- `residual_add_rms_norm_mul_fp4_sfa_fp16`
- `silu_mul_fp4_sfa_v2_fp16`
- `silu_mul_two_fp4_to_fp4`
- `silu_mul_two_mul_fp4_to_fp4`

Required comparisons:

| Scope | Baselines | Headline gate |
| --- | --- | --- |
| FP16-to-FP4 producers | FP16 math reference plus dequantized FP4/SFA output envelope | Correctness includes residual aliasing, dtype, p99 error, cosine, and explicit unsupported-shape rejection |
| v2 producer fast paths | v1 dequantized parity where v1 supports the shape | v2 is faster or equal on the published shape family |
| FP4-to-FP4 combiners | Dequantized FP4 reference and producer latency | Report as pipeline-continuity kernels unless a strong fused baseline exists |

Good public story:

- This package is not a standalone giant-speedup package. Its role is keeping
  FP4 runtime islands continuous by removing PyTorch elementwise breaks between
  low-bit GEMMs.
- Dequantization helpers are validation/debug APIs, not hot-path APIs.

### `fp4-gemm`

Primary kernels:

- `quantize_fp4_sfa_fp16`
- `dequantize_fp4_sfa_fp16`
- `fp4_w4a16_linear_bf16`

Required comparisons:

| Scope | Baselines | Headline gate |
| --- | --- | --- |
| W4A16 GEMM | PyTorch GEMM over the same dequantized FP4/SFA/SFB inputs, CUTLASS/cuBLASLt or FlashRT internal path when available | Kernel output must match the dequantized-input reference; variants are reported separately |
| FP4 producer helpers | Deterministic dequantized reference | Treated as validation/setup helpers unless included in an explicit fused model island |
| SM120 hardware scope | RTX 5090 source validation first, then Hub artifact validation and other Blackwell hardware | Keep CUDA 12.8+ Blackwell `sm_120a` label until broader source paths exist |

Good public story:

- This package is the native FP4 linear building block for transformer and
  diffuser demos that already keep activations and weights in packed FP4/SFA
  form.
- FP4-output GEMM variants stay internal until `can_implement` and public
  shape/tile validation are clean.

## Minimum Public Table Columns

Every public `RESULTS.md` table should include:

- package and kernel name;
- shape family and exact shape;
- selected tile or dispatch path;
- FlashRT latency;
- PyTorch eager latency and speedup;
- `torch.compile` latency and speedup, or a precise unsupported reason;
- strong library/unfused latency and speedup when required by this matrix;
- max error or byte-parity status;
- result label;
- GPU, driver, CUDA runtime, PyTorch version, warmup, and measured iterations.

## Execution Order

1. Run correctness and source-extension sweeps for every package.
2. Run PyTorch eager and `torch.compile` comparisons for every benchmark whose
   compiled reference first verifies against eager output.
3. Add unfused FlashRT/CUDA-chain baselines for fusion kernels where the source
   slices exist.
4. Add cuBLASLt/CUTLASS/other strong baselines for GEMM, FP4, FP8, and low-bit
   kernels.
5. Promote only rows labeled `headline` into top-level README charts.
6. Keep all tile sweeps and weak rows in `internal-tests/` until they are
   clearly labeled.
