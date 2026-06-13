# Source Sync

## Package Scope

`MiniMaxAI-msa-blackwell` is a decode-sparse Blackwell hardware-extension package for
MiniMax M3 sparse attention. It intentionally exposes only the decode paths that
FlashRT validated on GB10 / SM121.

FlashRT also validated this decode-sparse implementation inside the
MiniMax-Spark runtime on DGX Spark / GB10. The package here is the standalone
Hub-facing extraction of that validated model-path kernel.

The original MiniMaxAI Hub kernel is:

- <https://huggingface.co/kernels/MiniMaxAI/msa>

That package is SM100-only and is based on MiniMaxAI's CuTe-DSL MSA
implementation. This package provides a staged Blackwell port for consumer
Blackwell / GB10: a native CUDA helper for score-to-top-k block selection plus
the FlashRT-validated Triton decode sparse attention fallback.

## Upstream Provenance

Primary source copied from FlashRT `minimax-spark` validation work:

- FlashRT repository: `../official/FlashRT`
- Source commit: `3c38197`
- Commit subject: `P4: vendor SGLang MSA Triton kernels, decode-sparse validated on sm_121`
- Original upstream source: SGLang PR #27944, MiniMax-M3 sparse attention
  kernels under `python/sglang/srt/layers/attention/minimax_sparse_ops/`
- Original author attribution preserved in copied files:
  `Copyright 2025 XunhaoLai. All rights reserved.`
- License: Apache-2.0

Copied source files:

- `kernels/common/utils.py`
- `kernels/_compat.py`
- `kernels/decode/flash_with_topk_idx.py`
- `kernels/decode/topk_sparse.py`
- `kernels/naive/flash_with_topk_idx.py`
- `kernels/naive/topk_sparse.py`

FlashRT-added native source files:

- `csrc/msa_topk_from_scores.cu`
- `csrc/msa_topk_from_scores.cuh`
- `torch-ext/torch_binding.cpp`
- `torch-ext/torch_binding.h`
- `torch-ext/minimaxai_msa_blackwell/_native.py`

Package-local destination:

- `torch-ext/minimaxai_msa_blackwell/`

## Local Edits

- Renamed import package from the FlashRT staging name `kernels` to
  `minimaxai_msa_blackwell`.
- Kept the SGLang framework-coupling removal already done in FlashRT:
  `_compat.py` provides CUDA-only `is_hip()` and environment stubs.
- Public `__all__` exposes only validated decode/native APIs:
  `native_topk_from_scores`,
  `has_native_ops`,
  `flash_decode_with_topk_idx`,
  `flash_decode_with_gqa_share_sparse`,
  `naive_flash_decode_with_topk_idx`,
  `naive_flash_decode_with_gqa_share_sparse`.
- Prefill source is not shipped in this package because the current community
  request and FlashRT validation target are decode sparse attention on SM121.

## Runtime Contract

Common paged-KV layout:

```text
q            [batch, num_q_heads, head_dim] bf16/fp16
k_cache      [max_slots, num_kv_heads, head_dim] bf16/fp16
v_cache      [max_slots, num_kv_heads, head_dim] bf16/fp16
req_to_token [max_reqs, max_kv_len] int32
seq_lens     [batch] int32
slot_ids     [batch] int64
topk_idx     [num_kv_heads, batch, topk] int32, valid ids left-packed, -1 padded
score        [heads, batch, max_blocks] fp32 for native_topk_from_scores
```

MiniMax M3 validation shape:

```text
num_q_heads = 64
num_kv_heads = 4
head_dim = 128
block_size = 128
topk = 16
```

## Validation

FlashRT validated the decode sparse path on SM121 over context lengths
128 to 32768 with cosine similarity >= 0.999. The package-local test harness
uses standalone PyTorch references and does not require FlashRT, SGLang, or
vLLM.

The native score-to-top-k helper is checked against PyTorch `topk` set semantics
after artifact build. A local direct-extension compile smoke test on RTX 5090
also confirmed that the C++/CUDA binding compiles and returns exact top-k sets.

Model-path validation was done in the FlashRT `minimax-spark` work, where the
decode sparse MSA implementation is integrated into the MiniMax-Spark runtime
on DGX Spark / GB10. This is documented separately from package-local tests so
that the Hub package remains runtime-independent.
