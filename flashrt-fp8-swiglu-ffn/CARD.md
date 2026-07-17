# FlashRT FP8 GeGLU/SwiGLU FFN

This package provides Hugging Face Kernel Hub wrappers for FlashRT FP8 GeGLU/SwiGLU
FFN building blocks:

```text
FP8 input -> FP8 gate/up GEMM -> gate activation * up -> FP8 requant -> FP8 down GEMM -> BF16 output
```

It is intended for Gemma-style VLA/VLM language blocks and other FFN islands
where static FP8 activations and weights are already available.

## Kernels

- `fp8_gemm_bf16`: FP8 E4M3 GEMM with scalar input/weight scales and BF16
  output.
- `silu_mul_merged_quantize_fp8_static_bf16`: split merged BF16 gate/up output,
  compute `SiLU(gate) * up`, and requantize to FP8 E4M3.
- `gelu_mul_merged_quantize_fp8_static_bf16`: split merged BF16 gate/up output,
  compute `GELU_tanh(gate) * up`, and requantize to FP8 E4M3.
- `fp8_swiglu_mlp_bf16`: full FP8 SwiGLU MLP block with explicit optional
  scratch buffers.
- `fp8_geglu_mlp_bf16`: full FP8 GeGLU MLP block with explicit optional
  scratch buffers.
- `bf16_fp8_swiglu_mlp_bf16`: BF16 region entry for static-scale SwiGLU.
- `bf16_fp8_geglu_mlp_bf16`: BF16 region entry for static-scale GeGLU.

## When To Use

Use `fp8_geglu_mlp_bf16` for Gemma/PI0.5-style
`gelu_pytorch_tanh(gate) * up`. Use `fp8_swiglu_mlp_bf16` for true
`SiLU(gate) * up` blocks.

Use this package for static-shape model hot paths where the surrounding runtime
can keep FP8 tensors, weights, scales, and scratch buffers resident. Avoid
one-off calls between many unfused BF16 operations when reporting end-to-end
speedups; that measures Python/runtime boundaries instead of kernel value.
When the host block produces BF16, use a BF16 entry and preallocate its scratch
buffers to remove the standalone Python quantization boundary and support CUDA
Graph replay.

## Hardware

- CUDA 12.8+
- FP8-capable NVIDIA GPUs with cuBLASLt FP8 support
- ROCm artifacts currently target AMD CDNA3 `gfx942` and use
  `torch.float8_e4m3fnuz`

Current local CUDA validation is on RTX 5090. ROCm validation for this package
is scoped to the AMD `gfx942` FP8-FNUZ path. CDNA4/OCP-FP8 and RDNA targets are
not claimed by this package version.

## Notes

This package is a Tensor API integration layer. The upstream serving source of
truth remains FlashRT:

https://github.com/LiangSu8899/FlashRT

The BF16 region entries are traceable custom ops containing multiple launches,
not single-launch megakernels. Their production migration gate is bit-exact
against the package's established staged FP8 ops.
