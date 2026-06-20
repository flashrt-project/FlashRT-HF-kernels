# Source Sync

Synced from FlashRT:

- `official/FlashRT/csrc/kernels/gated_deltanet_qwen36.cu`
- `official/FlashRT/csrc/kernels/gated_deltanet_qwen36.cuh`

Public package rename:

- source file name: `gated_delta_attention`
- public API namespace: `flashrt/gated-delta-attention`

The kernel math is unchanged. Only the include name and Tensor API wrappers
were added for Kernel Hub packaging.
