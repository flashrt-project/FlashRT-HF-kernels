# FlashRT FP8 FFN

This package provides Hugging Face Kernel Hub wrappers for FlashRT FP8 FFN
building blocks. It is the first-choice package when replacing a full
`Linear -> GELU(tanh) -> Linear` FFN/MLP sublayer with static FP8 activations
and weights.

This package is model-agnostic. PI0.5, GROOT, VLA, VLM, and video-model usage
should call the same Tensor APIs rather than model-specific entry points.

## Kernels

- `fp8_gemm_bf16`: FP8 E4M3 GEMM with scalar input/weight scales and BF16
  output.
- `fp8_linear_bias_gelu_quant_bf16`: FP8 linear, BF16 bias, GELU(tanh), and
  FP8 requantization.
- `fp8_gelu_mlp_bf16`: full FP8 GELU MLP block:
  `FP8 up GEMM -> bias/GELU -> FP8 requant -> FP8 down GEMM -> bias`.

## When To Use

Use this package for model FFN islands where weights are already quantized and
activation/hidden scales are static for the benchmark or deployment slice.

Do not use it as a one-off Python call between many unfused BF16 operations if
the goal is end-to-end speed. For best results, keep FP8 tensors flowing across
adjacent FlashRT blocks and preallocate scratch buffers.

See the repository usage guide and replacement example for integration
patterns:
https://github.com/LiangSu8899/FlashRT-HF-kernels/blob/main/docs/usage.md
https://github.com/LiangSu8899/FlashRT-HF-kernels/blob/main/examples/replace_torch_ffn.py

## Hardware

- CUDA 12.8+
- FP8-capable NVIDIA GPUs with cuBLASLt FP8 support

Current local validation is on RTX 5090. Other hardware should be added to the
benchmark matrix before broader claims.

## Notes

This package is a Tensor API integration layer. The upstream serving source of
truth remains FlashRT. Shape-locked SM120 megakernels are intentionally not
included in this generic package.

The wrappers register fake/meta ops for `torch.compile` tracing. Benchmarks
only report `torch.compile` baselines when the compiled PyTorch reference is
verified equivalent to eager.
