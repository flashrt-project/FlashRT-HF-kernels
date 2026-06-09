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

## When To Use

Use `fp8_geglu_mlp_bf16` for Gemma/PI0.5-style
`gelu_pytorch_tanh(gate) * up`. Use `fp8_swiglu_mlp_bf16` for true
`SiLU(gate) * up` blocks.

Use this package for static-shape model hot paths where the surrounding runtime
can keep FP8 tensors, weights, scales, and scratch buffers resident. Avoid
one-off calls between many unfused BF16 operations when reporting end-to-end
speedups; that measures Python/runtime boundaries instead of kernel value.

## Hardware

- CUDA 12.8+
- FP8-capable NVIDIA GPUs with cuBLASLt FP8 support

Current local smoke validation is on RTX 5090. Full multi-hardware claims must
come from the repository validation matrix.

## Notes

This package is a Tensor API integration layer. The upstream serving source of
truth remains FlashRT:

https://github.com/LiangSu8899/FlashRT
