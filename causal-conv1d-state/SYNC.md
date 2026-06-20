# Source Sync

Synced from FlashRT:

- `official/FlashRT/csrc/kernels/causal_conv1d_qwen36.cu`
- `official/FlashRT/csrc/kernels/causal_conv1d_qwen36.cuh`

Public package rename:

- source file name: `causal_conv1d_state`
- public API namespace: `flashrt/causal-conv1d-state`

The kernel math is unchanged. Only the include name and Tensor API wrappers
were added for Kernel Hub packaging.
