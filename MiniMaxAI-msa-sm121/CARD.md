---
library_name: kernels
license: apache-2.0
tags:
  - cuda
  - triton
  - minimax
  - sparse-attention
  - blackwell
---

# MiniMaxAI MSA SM121

This is a FlashRT-packaged SM121 hardware extension for MiniMaxAI MSA sparse
attention. The original MiniMaxAI Hub kernel package is
<https://huggingface.co/kernels/MiniMaxAI/msa>, which targets SM100.

This package exposes standalone PyTorch/Triton Tensor APIs for MiniMax M3 sparse
attention decode paths. It is intended for GB10 / consumer
Blackwell validation and integration experiments where SM100-only kernels are
not available.

FlashRT validated the same decode-sparse implementation inside the
MiniMax-Spark model runtime on DGX Spark / GB10. The public package is the
standalone Hub-facing Tensor API extracted from that validated model path.

Validation focus:

- MiniMax M3 decode sparse attention shapes: Hq=64, Hkv=4, D=128, block=128,
  topk=16.
- Context lengths: 128, 2048, 4096, 32768.
- Correctness threshold: cosine similarity >= 0.999 against paged FP32 PyTorch
  references.
- Model-path validation: MiniMax-Spark decode sparse attention integration in
  FlashRT on DGX Spark / GB10.

Source provenance and local edits are documented in `SYNC.md`.
