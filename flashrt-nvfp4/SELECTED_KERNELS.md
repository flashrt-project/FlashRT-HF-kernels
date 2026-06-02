# Selected Kernels

`flashrt-nvfp4` is the Blackwell low-bit package. It should expose reusable
NVFP4 layout and fused GEMM surfaces, not model-specific serving paths.

## Tier 1: Layout Helpers

| Public API | FlashRT source | Reason |
| --- | --- | --- |
| `nvfp4_sf_linear_to_swizzled` | `official/FlashRT/csrc/quantize/nvfp4_sf_reshape_sm120.cu` | Converts linear NVFP4 scale factors into the swizzled layout expected by downstream GEMM and quant paths. |
| `nvfp4_sf_swizzled_bytes` | `official/FlashRT/csrc/quantize/nvfp4_sf_reshape_sm120.cu` | Lets callers allocate exact SFA/SFB buffers without importing FlashRT internals. |

Promotion target: first buildable slice for this package.

## Tier 2: Fused NVFP4 GEMM Epilogues

| Public API | FlashRT source | Reason |
| --- | --- | --- |
| `nvfp4_linear_bias_gelu_fp4out_sm120` | `official/FlashRT/csrc/gemm/fp4/cutlass_nvfp4_gemm_bias_gelu_fp4out_sm120.cu` | Fuses NVFP4 GEMM, bias, GELU, and FP4 output quantization. This is the strongest functional story in the package. |
| `nvfp4_linear_bias_gelu_bf16out_sm120` | `official/FlashRT/csrc/gemm/fp4/cutlass_nvfp4_gemm_bias_gelu_bf16out_sm120.cu` | Fuses NVFP4 GEMM with bias+GELU when the next op still needs BF16. |
| `nvfp4_linear_streamk_bias_bf16out_sm120` | `official/FlashRT/csrc/gemm/fp4/cutlass_nvfp4_gemm_dn_streamk_bias_sm120.cu` | Stream-K down-projection path for large-K/large-N low-bit decode and prefill shapes. |

Promotion target: after CUTLASS dependency isolation and benchmark parity.

## Hardware Scope

Initial support is SM120/SM120a only. Do not claim broader CUDA architecture
support until the kernels are built and measured on those architectures.

## Validation Grid

- Layout helpers: multiple `M,N` pairs, odd row counts, alignment boundaries,
  byte-size checks, round-trip layout reference.
- GEMM epilogues: `M in {1, 4, 8, 16, 32, 128}`, K/N families from VLA and
  Transformer FFN shapes, BF16 and FP4-output reference paths.
- Baselines: CUTLASS unfused GEMM plus PyTorch/CUDA elementwise epilogue, and
  FlashRT internal output parity.
