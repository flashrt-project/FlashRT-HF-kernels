---
library_name: kernels
tags:
- cuda
- pytorch
- flashrt
- causal-conv1d
- qwen3
- transformer
---

# Causal Conv1D State

BF16 causal depthwise Conv1D and state-update kernels from FlashRT, packaged
for Hugging Face Kernel Hub. This package is useful for transformer runtimes
that keep Conv1D state on device during decode/verify/prefill.

See `README.md` for the public API and examples.
