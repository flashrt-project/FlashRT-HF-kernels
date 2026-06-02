# Tile And Shape Coverage

This document defines the v1 batch tuning standard. A kernel is not ready for
the first public v1 batch until it has a documented shape grid, tile sweep,
correctness baseline, and performance baseline.

## Coverage Levels

| Level | Requirement | Use |
| --- | --- | --- |
| S0 | One smoke shape passes correctness | Source sync only |
| S1 | Decode, small-M, prefill, and wide-dim representative shapes pass | Draft package |
| S2 | Full first-batch grid passes correctness and benchmark reporting | Buildable package |
| S3 | Tile sweep chooses a default policy and documents rejected variants | Release candidate |
| S4 | Same claims verified on every advertised CUDA architecture | Community promotion |

## First-Batch Shape Grid

### FP8 Quant Epilogues

Applies to:

- `bias_gelu_quantize_fp8_static_bf16`
- `gelu_quantize_fp8_static_bf16`
- `channel_scale_quantize_fp8_static_bf16`

| Family | M | N |
| --- | --- | --- |
| Decode | 1, 2, 4, 8 | 4096 |
| Small batch | 16, 32 | 4096 |
| Prefill | 64, 128, 256 | 4096 |
| Wide hidden | 16, 128 | 8192 |
| VLA/video FFN | 16, 64 | 12288, 16384 |

Tile variables:

- `FLASHRT_QUANT_BLOCK_SIZE in {128, 256, 512, 1024}`.
- Default policy must be chosen per operation family, not only one global
  number, if the sweep shows a consistent split.

Current SM120 default policy:

- `bias_gelu_quantize_fp8_static_bf16`: 512 for biased decode micro-batches,
  1024 for `M <= 32` with `N <= 8192`, 512 for biased VLA-width `M <= 32`,
  otherwise 256.
- `gelu_quantize_fp8_static_bf16`: 256 for `M = 1` and VLA-width cases, 1024
  for `2 <= M <= 32` with `N <= 8192`, otherwise 256.
- `channel_scale_quantize_fp8_static_bf16`: 1024 for `M <= 32` with
  `N <= 8192`, 512 for 4096-wide `M <= 128`, otherwise 256.

Promotion threshold:

- Correctness: exact FP8 byte parity with the PyTorch fake-quant reference.
- Performance: at least 4x over PyTorch eager for headline shapes, or a clear
  memory-bandwidth explanation when a shape is kept for completeness.

### VLA And Video QKV Post-Processing

Applies to:

- `q_norm_rope_bf16`
- `k_norm_rope_v_cache_bf16`
- `qkv_split_norm_rope_bf16`

| Family | Shape |
| --- | --- |
| Decode rows | heads in 1, 4, 8, 16, 24, 32, 48; head_dim 128 |
| Short context | batch 1, tokens 1, 4, 16; heads 24; head_dim 128 |
| Vision/video tokens | batch 1, tokens 64, 256, 1024, 2520, 4096; heads 24; head_dim 128 |
| Head-count sweep | batch 1, tokens 64; heads 8, 16, 24, 32, 48; head_dim 128 |

Tile variables:

- `FLASHRT_QKV_ROPE_BLOCK_SIZE in {128, 256, 512}`.
- Current SM120 default is 512-thread CTAs for `tokens <= 64` and 256-thread
  CTAs otherwise.

Promotion threshold:

- Correctness: BF16-level tolerance against PyTorch RMSNorm + RoPE reference.
- Performance: 20x+ over PyTorch eager for headline video-token shapes.

### NVFP4 Layout And GEMM Epilogues

Applies to:

- `nvfp4_sf_linear_to_swizzled`
- `nvfp4_linear_bias_gelu_fp4out_sm120`
- `nvfp4_linear_bias_gelu_bf16out_sm120`
- `nvfp4_linear_streamk_bias_bf16out_sm120`

| Family | Shape |
| --- | --- |
| Layout boundary | rows 1, 2, 31, 32, 33, 127, 128, 129 |
| Layout contracted dim | D 1024, 2048, 4096, 8192, 12288, 16384 |
| GEMM small-M | M 1, 4, 8, 16, 32; K/N from 4096, 8192, 12288, 16384 |
| GEMM prefill | M 64, 128, 256; K/N from 4096, 8192, 12288, 16384 |

Tile variables:

- Layout helper: block shapes `(8,32)`, `(16,16)`, `(32,8)` before freezing.
- CUTLASS epilogues: scheduler, persistent/stream-K choice, and epilogue
  visitor path must be documented per source file.

Promotion threshold:

- Correctness: byte-for-byte layout parity and fake-quant tolerance for FP4
  output paths.
- Performance: compare against unfused CUTLASS/cuBLAS plus separate epilogue
  kernels, not only PyTorch eager.

### Small-M GEMM

Applies to:

- `nvfp4_w4a4_decode_matvec_bf16out`
- `nvfp4_w4a4_smallm_warpsplit_bf16out`
- `tiny_fp8_smallm_gemm_bf16out`

| Family | Shape |
| --- | --- |
| Decode | M 1; K/N from 4096, 8192, 12288, 16384 |
| Small batch | M 2, 4, 8, 16, 32; K/N from 4096, 8192, 12288 |
| Tiny FP8 fixed | every fixed family exported by the source binding |

Tile variables:

- Rows per CTA, warps per CTA, K split, and rows per warp.
- Exact dispatch boundaries must be measured; do not expose a generic
  dispatcher before the boundary grid is stable.

Promotion threshold:

- Correctness: dequantized PyTorch reference and unsupported-shape rejection.
- Performance: compare against cuBLASLt/CUTLASS low-bit paths where available.

### Fused Quantization

Applies to:

- `silu_mul_quant_nvfp4_swizzled_bf16`
- `silu_mul_merged_quant_nvfp4_swizzled_bf16`
- `rmsnorm_quant_nvfp4_sfa_fp16`
- `residual_rmsnorm_quant_nvfp4_sfa_fp16`

| Family | Shape |
| --- | --- |
| Decode FFN | rows 1, 2, 4, 8; hidden 4096, 8192, 12288, 16384 |
| Small batch FFN | rows 16, 32; hidden 4096, 8192, 12288, 16384 |
| Prefill/video | rows 64, 128, 256, 1024, 2520; hidden 4096, 8192, 12288 |

Tile variables:

- Rows per CTA, columns per CTA, scale blocks per CTA, and shared-memory use.
- Split and merged gate/up variants must be measured separately.

Promotion threshold:

- Correctness: fake-quant reference and residual aliasing tests.
- Performance: report both latency and effective memory bandwidth.
