# Source Sync

Synced from `official/FlashRT`:

- FP8 cuBLASLt descale GEMM convention and descriptor cache used by the
  production serving paths.
- BF16 `SiLU(gate) * up -> FP8 E4M3` merged quantization logic from the FlashRT
  BF16 kernel path.

Local adaptation:

- Raw pointer APIs were converted to Tensor-based `torch.ops` bindings.
- The public API is model-agnostic and uses merged gate/up row layout
  `[gate | up]`.
- Scratch buffers are explicit optional arguments so model runtimes can avoid
  repeated allocation in hot paths.
- The package keeps per-tensor scales only. Block-scaled FP8 and shape-locked
  megakernels should be reviewed as separate packages.
