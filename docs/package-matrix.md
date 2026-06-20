# Package Matrix

| Package | First public APIs | Upstream FlashRT areas | Main baseline | Showcase role |
| --- | --- | --- | --- | --- |
| `flashrt-gemm-epilogues` | `bias_gelu_quantize_fp8_static_bf16`, `gelu_quantize_fp8_static_bf16`, `channel_scale_quantize_fp8_static_bf16`, selected `bf16_gemm_bias*` | `csrc/gemm`, selected quant epilogues | `torch.addmm` plus elementwise/quant ops | First buildable package; FP8 quant epilogue headline |
| `flashrt-fp8-ffn` | `fp8_gemm_bf16`, `fp8_linear_bias_gelu_quant_bf16`, `fp8_gelu_mlp_bf16` | `csrc/kernels/decoder_fused.cu`, FP8 GEMM descale paths, FP8 quant epilogues | PyTorch eager full-FFN reference plus segmented compile-stable `torch.compile` reference | Full VLA/VLM FFN sublayer showcase |
| `fp8-gemm` | `fp8_linear_bf16`, `fp8_linear_residual_bf16` | `csrc/gemm/fp8_gemv_m1_sm120.*`, `csrc/gemm/fp8_smallM_handtuned*` | PyTorch eager and compile-stable FP8 reference | Native Blackwell FP8 decode/small-M linear showcase |
| `flashrt-vla-video` | `q_norm_rope_bf16`, `k_norm_rope_v_cache_bf16`, `qkv_split_norm_rope_bf16`, then `residual_rmsnorm_quant_nvfp4`, `silu_mul_quant_nvfp4`, `video_conv_lowbit` | `csrc/kernels`, `flash_wm/csrc`, `csrc/conv`, `csrc/quantize` | PyTorch eager, FlashRT internal reference, model-block baseline | First 20-30x+ showcase package |
| `flashrt-nvfp4` | `nvfp4_sf_linear_to_swizzled`, `nvfp4_sf_swizzled_bytes`; planned `nvfp4_linear_bias_gelu_fp4out_sm120`, `nvfp4_linear_bias_gelu_bf16out_sm120`, `nvfp4_linear_streamk_bias_bf16out_sm120` | `csrc/quantize`, `csrc/gemm/fp4`, `flash_wm/csrc` | CUTLASS/cuBLAS where applicable, PyTorch dequant reference | Buildable NVFP4 layout helper; strong Blackwell low-bit showcase follows |
| `flashrt-smallm-gemm` | `nvfp4_w4a4_decode_matvec_bf16out`, `nvfp4_w4a4_smallm_warpsplit_bf16out`, `tiny_fp8_smallm_gemm_bf16out` | `csrc/kernels`, `csrc/kernels/megakernel`, small-M matvec/matmul files | cuBLASLt, generic CUTLASS, PyTorch eager | Decode latency showcase for LLM/VLA serving |
| `flashrt-fused-quant` | `silu_mul_quant_nvfp4_swizzled_bf16`, `silu_mul_merged_quant_nvfp4_swizzled_bf16`, `residual_rmsnorm_quant_nvfp4_swizzled_bf16`, `rmsnorm_quant_nvfp4_sfa_fp16` | `csrc/kernels`, `csrc/quantize`, `csrc/fused_fp4`, `flash_wm/csrc` | PyTorch eager reference | Shared utility package; useful when split from model-specific stacks |
| `fp4-fused-ops` | `rms_norm_fp4_sfa_fp16`, `residual_add_rms_norm_fp4_sfa_v2_fp16`, `silu_mul_fp4_sfa_v2_fp16`, `silu_mul_two_fp4_to_fp4` | `csrc/fused_fp4`, `csrc/quantize` | FP16 math reference, v1/v2 dequantized parity, producer latency | Native Blackwell FP4 producer/combiner package for continuous low-bit runtime paths |
| `fp4-gemm` | `quantize_fp4_sfa_fp16`, `dequantize_fp4_sfa_fp16`, `fp4_w4a16_linear_bf16` | `csrc/gemm/fp4`, `csrc/quantize` | PyTorch GEMM over the same dequantized FP4/SFA/SFB inputs | Native Blackwell NVFP4 W4A16 linear package |
| `fp8-kv-attention` | `xqa_bf16_fp8kv`, `causal_spec_mask`, `default_page_table`, `allocate_workspace` | `csrc/attention/flashinfer_xqa_src`, `csrc/kernels/qwen36_flashinfer_xqa.*` | PyTorch FP8-dequant attention reference with the same speculative mask | Direct BF16-Q + FP8-KV XQA package for Qwen3.6-style decode/verify |
| `causal-conv1d-state` | `causal_conv1d_bf16`, `causal_conv1d_update_bf16`, `causal_conv1d_update_inout_bf16`, `causal_conv1d_update_chunk_parallel_gqa_bf16` | `csrc/kernels/causal_conv1d_qwen36.*` | PyTorch BF16 causal Conv1D state reference | Qwen3.6-style pre-linear-attention Conv1D state update and GQA split |
| `gated-delta-attention` | `gated_delta_recurrent_bf16`, `gated_delta_recurrent_inout_bf16`, `gated_delta_recurrent_f32state_bf16io`, `gated_delta_chunk_bf16`, `gated_delta_chunk_smem_bf16` | `csrc/kernels/gated_deltanet_qwen36.*` | PyTorch Gated DeltaNet recurrent/chunk reference with BF16 state contract | Stateful linear-attention recurrence for transformer decode/verify/prefill |

## V1 Batch Blocks

The first release is a batch release. These blocks are peers; the order below
is not a priority order.

| V1 block | Packages | Message |
| --- | --- | --- |
| FP8/GEMM and FFN | `flashrt-gemm-epilogues`, `flashrt-fp8-ffn` | FP8 quant epilogues plus full FP8 GELU MLP/FFN sublayers |
| VLA/video post-processing | `flashrt-vla-video` | 19-40x local RTX 5090 evidence for fused QKV/norm/RoPE/cache paths |
| Blackwell NVFP4/FP4 low-bit | `flashrt-nvfp4`, `flashrt-smallm-gemm` | Layout helpers, fused low-bit GEMM epilogues, and small-M/decode kernels |
| Fused quantization | `flashrt-fused-quant` | Activation, residual, norm, and low-bit quantization fusion |
| Native FP4 runtime path | `fp4-fused-ops`, `fp4-gemm` | FP16-to-NVFP4 producers, FP4-to-FP4 combiners, and NVFP4 W4A16 GEMM |
| FP8 KV attention | `fp8-kv-attention` | BF16-query XQA over FP8 E4M3 paged K/V cache for long-context decode/verify |
| Qwen3.6 linear-attention state | `causal-conv1d-state`, `gated-delta-attention`, `linear-attention-primitives` | Conv1D state update, Gated DeltaNet recurrence/chunks, and staging helpers |

Do not run full builder packaging for one block while the other v1 blocks are
still missing source-extension tests, benchmark grids, or examples. Full
builder work is reserved for the v1 release window.

## Evidence Levels

| Level | Requirement | Use |
| --- | --- | --- |
| L0 | Source-extension import and correctness pass locally | Development |
| L1 | Microbenchmark wins against fair PyTorch/library baselines | Public package docs |
| L2 | Model-block benchmark shows a meaningful latency or bandwidth win | Showcase candidate |
| L3 | End-to-end HF-style model example shows throughput or latency benefit | First-update headline |
| L4 | Multi-hardware validation covers the claimed CUDA architectures | Community promotion |

## Naming Policy

Use names that describe math and data movement. Avoid model names in package and
API names unless the function cannot be made generic.

Good:

- `fp8_linear_bias_gelu`
- `residual_rmsnorm_quant`
- `smallm_nvfp4_gemm`
- `patch_embed_bias_pos`

Avoid:

- `qwen36_*`
- `pi05_*`
- `groot_*`
- `motus_*`

Model names may appear in benchmark shape labels or documentation explaining
provenance.

## Hardware Policy

Build success is not the same as supported hardware. Each package should state
the CUDA architectures that are validated, and SM-specific kernels should not be
advertised as broader than they are.

Initial targets:

- FP8 quant epilogues: validate across the CUDA/PyTorch builder matrix and add
  runtime checks on more GPUs when available.
- Blackwell NVFP4/FP4 kernels: label as SM120/SM120a until other architectures
  are implemented and measured.
- Native FP4 producer/GEMM packages: label as CUDA 12.8+ Blackwell
  `sm_120a` until non-Blackwell source paths are added and measured.
- FP8 KV attention: v1 artifact target is CUDA 12.8+ Blackwell `sm_120a`
  fixed-shape XQA. Add `sm_121` only after a matching artifact build and
  runtime validation pass on that architecture.
- H100/SM90 paths: package separately or gate explicitly if the implementation
  differs from the SM120 path.
