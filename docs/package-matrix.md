# Package Matrix

| Package | First public APIs | Upstream FlashRT areas | Main baseline | Showcase role |
| --- | --- | --- | --- | --- |
| `flashrt-gemm-epilogues` | `bias_gelu_quantize_fp8_static_bf16`, `gelu_quantize_fp8_static_bf16`, `channel_scale_quantize_fp8_static_bf16`, selected `bf16_gemm_bias*` | `csrc/gemm`, selected quant epilogues | `torch.addmm` plus elementwise/quant ops | First buildable package; FP8 quant epilogue headline |
| `flashrt-vla-video` | `q_norm_rope_bf16`, `k_norm_rope_v_cache_bf16`, `qkv_split_norm_rope_bf16`, then `residual_rmsnorm_quant_nvfp4`, `silu_mul_quant_nvfp4`, `video_conv_lowbit` | `csrc/kernels`, `flash_wm/csrc`, `csrc/conv`, `csrc/quantize` | PyTorch eager, FlashRT internal reference, model-block baseline | First 20-30x+ showcase package |
| `flashrt-nvfp4` | `quantize_nvfp4`, `dequantize_nvfp4`, `nvfp4_linear`, `nvfp4_linear_bias_gelu`, `reshape_sfa`, `sfa_size_bytes` | `csrc/quantize`, `csrc/gemm/fp4`, `flash_wm/csrc` | CUTLASS/cuBLAS where applicable, PyTorch dequant reference | Strong Blackwell low-bit showcase; hardware scope must be explicit |
| `flashrt-smallm-gemm` | `smallm_fp8_gemm`, `smallm_nvfp4_gemm`, `splitk_decode_gemv`, `smallm_bf16_matmul` | `csrc/gemm/fp8_smallM*`, small-M matvec/matmul files | cuBLASLt, generic CUTLASS, PyTorch eager | Decode latency showcase for LLM/VLA serving |
| `flashrt-fused-quant` | `rmsnorm_quant`, `residual_rmsnorm_quant`, `swiglu_quant`, `qkv_rope_split` | `csrc/kernels`, `csrc/quantize`, `csrc/fused_fp4` | PyTorch eager reference | Shared utility package; useful when split from model-specific stacks |

## Showcase Priority

1. `flashrt-gemm-epilogues`: ship first because the format is proven. Public
   messaging should emphasize FP8 quant epilogues, not shape-sensitive BF16 GEMM
   epilogues.
2. `flashrt-vla-video`: strongest first showcase. The implemented Q/K
   post-processing slice has internal 20-30x+ benchmark evidence across
   decode and video-token packed-QKV paths.
3. `flashrt-nvfp4`: strongest Blackwell low-bit story. Keep hardware support
   honest; SM120-only kernels should be labeled that way.
4. `flashrt-smallm-gemm`: strong serving story once benchmarks beat cuBLASLt or
   generic CUTLASS on the right decode shapes.
5. `flashrt-fused-quant`: useful as a dependency-like package, but less
   attention-grabbing unless tied to a model path.

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
- H100/SM90 paths: package separately or gate explicitly if the implementation
  differs from the SM120 path.
