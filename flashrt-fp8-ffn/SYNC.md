# Source Sync

Synced from `official/FlashRT`:

- `csrc/kernels/decoder_fused.cu`: FP8 cuBLASLt descale GEMM convention and
  descriptor cache.
- `csrc/quantize/bias_gelu_quantize_fp8.*`: bias/GELU/static FP8 quant math.

Local adaptation:

- Raw pointer APIs were converted to Tensor-based `torch.ops` bindings.
- The MLP block composes existing production primitives into a Hub-loadable
  FFN surface.
- The package keeps per-tensor scales only. Block-scaled FP8 and shape-locked
  megakernels should be reviewed as separate packages.
