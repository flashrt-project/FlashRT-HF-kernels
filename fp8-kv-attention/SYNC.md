# Source Sync

Source of truth:

- `official/FlashRT/csrc/attention/flashinfer_xqa_src/`
- `official/FlashRT/csrc/kernels/qwen36_flashinfer_xqa.cu`
- `official/FlashRT/csrc/kernels/qwen36_flashinfer_xqa.cuh`

Public package changes:

- Public launch symbol renamed to `flashrt_xqa_bf16_fp8kv`.
- Fixed XQA compile definitions are held in `csrc/xqa_mha_configured.cu`.
- Python/C++ boundary exposes Tensor APIs and shape checks instead of FlashRT
  serving-internal pointer APIs.

When syncing from FlashRT, preserve the fixed v1 shape contract unless the Hub
package version is bumped and validation is expanded.
