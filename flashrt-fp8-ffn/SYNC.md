# Source Sync

Upstream FlashRT audit commits:

- Base FP8 FFN audit: `077c06372ef8ed5f86dfdeb6eaa0b4c0777ebc15`.
- Vectorized fused-producer audit: `67da1a2589da8b53b5369e0515a7f5a06b97e650`.

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
- Four-element-aligned BF16 input quantization, bias/GELU/FP8 production, and
  final BF16 bias use vectorized BF16 loads and packed FP8 stores. This adapts
  the producer pattern from the newer FlashRT fused FFN path to the generic
  BF16 Tensor API. Non-four-aligned dimensions retain the scalar path.
- RTX 5090 tile probes rejected the package-local hand-tuned alternatives
  because every tested mid-M case was slower than the retained cuBLASLt path.
- The package keeps per-tensor scales only. Block-scaled FP8 and shape-locked
  megakernels should be reviewed as separate packages.
