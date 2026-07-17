# Source Sync

Upstream FlashRT audit commit: `077c06372ef8ed5f86dfdeb6eaa0b4c0777ebc15`.

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
- BF16 region entries add package-local production-order input quantization
  and optional zero-row padding while retaining the established FP8 GEMMs and
  SwiGLU/GeGLU epilogues.
- RTX 5090 tile probes rejected slower package-local hand-tuned mid-M GEMMs;
  the public path retains the measured cuBLASLt implementation.
- The package keeps per-tensor scales only. Block-scaled FP8 and shape-locked
  megakernels should be reviewed as separate packages.
