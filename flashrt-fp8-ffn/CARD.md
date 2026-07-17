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
- `bf16_fp8_gelu_mlp_bf16`: BF16 region entry that performs static input
  quantization and the full FP8 GELU MLP behind one traceable custom-op call.

## When To Use

Use this package for model FFN islands where weights are already quantized and
activation/hidden scales are static for the benchmark or deployment slice.
Use the BF16 entry when the surrounding Transformers/Diffusers block naturally
produces BF16 and the old Python-side quantization call is an integration seam.

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
- ROCm artifacts currently target AMD CDNA3 `gfx942` and use
  `torch.float8_e4m3fnuz`

Current local CUDA validation is on RTX 5090. ROCm validation for this package
is scoped to the AMD `gfx942` FP8-FNUZ path. CDNA4/OCP-FP8 and RDNA targets are
not claimed by this package version.

The BF16-entry API in this revision is published in the current x86_64 CUDA
and ROCm build matrix. An older aarch64 directory retained by the incremental
Hub repository predates this API and is not a BF16-entry compatibility claim.
Build and validate a matching aarch64 variant before enabling this API there.

## Notes

This package is a Tensor API integration layer. The upstream serving source of
truth remains FlashRT. Shape-locked SM120 megakernels are intentionally not
included in this generic package.

The wrappers register fake/meta ops for `torch.compile` tracing. Benchmarks
only report `torch.compile` baselines when the compiled PyTorch reference is
verified equivalent to eager.

For a static hot path, preallocate the optional FP8/BF16 scratch tensors and
capture the call with CUDA Graph. The BF16 entry is a region API containing
multiple launches; it is not advertised as a single-launch megakernel.

Four-element-aligned dimensions automatically use vectorized BF16 producers
and packed FP8 stores. Non-aligned dimensions retain the scalar implementation.
Correctness promotion requires exact output parity with the established staged
FlashRT path; comparison with an independent BF16 model is reported separately
as quantization quality rather than migration parity.
