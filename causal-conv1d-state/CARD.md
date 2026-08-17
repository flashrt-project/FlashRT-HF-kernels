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

Version 2 adds `causal_conv1d_update_steps_gqa_bf16`, a step-batched prefill
entry for `C=10240`, `K=4` that emits flattened 16/16/48-head GQA tensors and
updates the three-row input state in place.
