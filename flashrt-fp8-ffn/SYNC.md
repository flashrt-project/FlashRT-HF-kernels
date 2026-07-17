# Source Sync

Upstream FlashRT audit commit: `077c06372ef8ed5f86dfdeb6eaa0b4c0777ebc15`.

Synced from `official/FlashRT`:

- `csrc/kernels/decoder_fused.cu`: FP8 cuBLASLt descale GEMM convention and
  descriptor cache.
- `csrc/quantize/bias_gelu_quantize_fp8.*`: bias/GELU/static FP8 quant math.

Local adaptation:

- Raw pointer APIs were converted to Tensor-based `torch.ops` bindings.
- The MLP block composes existing production primitives into a Hub-loadable
  FFN surface.
- The BF16 region entry adds package-local static input quantization with the
  production reciprocal-multiply arithmetic order and optional zero-row
  padding. It preserves the established FP8 GEMM/epilogue implementation.
- RTX 5090 tile probes rejected the package-local hand-tuned alternatives
  because every tested mid-M case was slower than the retained cuBLASLt path.
- The package keeps per-tensor scales only. Block-scaled FP8 and shape-locked
  megakernels should be reviewed as separate packages.
