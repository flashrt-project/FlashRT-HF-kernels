---
library_name: kernels
license: apache-2.0
tags:
  - cuda
  - triton
  - native-cuda
  - minimax
  - sparse-attention
  - blackwell
---

# MiniMaxAI MSA Blackwell

FlashRT-packaged Blackwell-family extension for MiniMaxAI MSA decode sparse
attention. The upstream MiniMaxAI package is
[`MiniMaxAI/msa`](https://huggingface.co/kernels/MiniMaxAI/msa), which targets
SM100. This package keeps the MiniMax MSA Tensor API style and extends the
decode-sparse path to CUDA compute capability 12.x Blackwell targets.

The package is validated in the FlashRT MiniMax-Spark runtime on DGX Spark /
GB10 and exposes standalone Tensor APIs for use from Python.

## Install and Load

```python
from kernels import get_kernel

msa = get_kernel(
    "flashrt/MiniMaxAI-msa-blackwell",
    version=1,
    trust_remote_code=True,
)
```

## Current v1 Functions

This v1 Blackwell package intentionally exposes the subset that has already
been ported and validated on FlashRT's MiniMax-Spark decode path. It is not yet
a drop-in mirror of every public function exported by the upstream
[`MiniMaxAI/msa`](https://huggingface.co/kernels/MiniMaxAI/msa) SM100 package.

### `has_native_ops() -> bool`

Returns whether the native CUDA helper extension was loaded.

```python
assert msa.has_native_ops()
```

### `native_topk_from_scores(score, seq_lens, block_size, topk) -> topk_idx`

Native CUDA helper that converts precomputed MiniMax MSA block scores to sparse
block ids.

- `score`: CUDA `float32`, shape `[heads, batch, max_blocks]`
- `seq_lens`: CUDA `int32`, shape `[batch]`
- `block_size`: currently validated with `128`
- `topk`: currently validated with `16`
- return: CUDA `int32`, shape `[heads, batch, topk]`

```python
import torch

score = torch.randn(64, 1, 256, device="cuda", dtype=torch.float32)
seq_lens = torch.tensor([32768], device="cuda", dtype=torch.int32)
topk_idx = msa.native_topk_from_scores(score, seq_lens, block_size=128, topk=16)
```

### `flash_decode_with_topk_idx(...) -> (index_value, topk_idx, real_seq_lens)`

MiniMax decode indexer. In the common sparse-indexing mode, pass
`disable_index_value=True` and the function returns `(None, topk_idx,
real_seq_lens)`.

Typical indexer inputs:

- `q`: CUDA `bfloat16`, shape `[batch, 1, 128]`
- `k_cache`: CUDA `bfloat16`, shape `[ctx, 1, 128]`
- `req_to_token`: CUDA `int32`, shape `[batch, ctx]`
- `seq_lens`: CUDA `int32`, shape `[batch]`
- `slot_ids`: CUDA integer, shape `[batch]`
- output `topk_idx`: CUDA `int32`, shape `[1, batch, topk]`

### `flash_decode_with_gqa_share_sparse(...) -> out`

Block-sparse GQA decode attention. This consumes `topk_idx` from the indexer
and applies the selected sparse blocks to the full GQA attention path.

Typical MiniMax M3 decode inputs:

- `q`: CUDA `bfloat16`, shape `[batch, 64, 128]`
- `k_cache`: CUDA `bfloat16`, shape `[ctx, 4, 128]`
- `v_cache`: CUDA `bfloat16`, shape `[ctx, 4, 128]`
- `topk_idx`: CUDA `int32`, shape `[4, batch, 16]`
- return: CUDA `bfloat16`, shape `[batch, 64, 128]`

### Reference Functions

These are provided for local correctness checks:

- `naive_flash_decode_with_topk_idx(...)`
- `naive_flash_decode_with_gqa_share_sparse(...)`

## Upstream API Compatibility Plan

The upstream `MiniMaxAI/msa` package currently exposes a broader SM100 API
surface:

- `sparse_atten_func`
- `sparse_atten_nvfp4_kv_func`
- `sparse_decode_atten_func`
- `SparseDecodePagedAttentionWrapper`
- `fp4_indexer_block_scores`
- `build_k2q_csr`
- `SparseK2qCsrBuilderSm100`
- `Nvfp4QuantizedTensor`
- `quantize_bf16_to_nvfp4_128x4`
- `quantize_kv_bf16_to_nvfp4_128x4`
- `dequantize_nvfp4_128x4_to_bf16`
- `swizzle_nvfp4_scale_to_128x4`
- `nvfp4_global_scale_from_amax`

Those names are not advertised as available in this v1 package unless they are
listed in the previous section. The v2 goal is to add a compatibility layer for
the official MiniMaxAI API names where the Blackwell implementation is ready,
starting with the decode path (`sparse_decode_atten_func` and
`SparseDecodePagedAttentionWrapper`), then expanding to indexing/CSR and NVFP4
helpers after separate correctness validation.

## Minimal Decode Example

```python
import torch
from kernels import get_kernel

msa = get_kernel("flashrt/MiniMaxAI-msa-blackwell", version=1, trust_remote_code=True)

batch, hq, hkv, d = 1, 64, 4, 128
ctx, block, topk = 4096, 128, 16
device, dtype = "cuda", torch.bfloat16

q = torch.randn(batch, hq, d, device=device, dtype=dtype)
k_cache = torch.randn(ctx, hkv, d, device=device, dtype=dtype)
v_cache = torch.randn(ctx, hkv, d, device=device, dtype=dtype)
q_index = torch.randn(batch, 1, d, device=device, dtype=dtype)
k_index = torch.randn(ctx, 1, d, device=device, dtype=dtype)
req_to_token = torch.arange(ctx, device=device, dtype=torch.int32).view(1, -1)
seq_lens = torch.tensor([ctx], device=device, dtype=torch.int32)
slot_ids = torch.zeros(batch, device=device, dtype=torch.int64)

_, topk_idx, _ = msa.flash_decode_with_topk_idx(
    q_index,
    None,
    k_index,
    None,
    req_to_token,
    seq_lens,
    max_seqlen=ctx,
    slot_ids=slot_ids,
    block_size=block,
    topk=topk,
    init_blocks=0,
    local_blocks=1,
    disable_index_value=True,
)

out = msa.flash_decode_with_gqa_share_sparse(
    q,
    None,
    k_cache,
    v_cache,
    req_to_token,
    seq_lens,
    slot_ids,
    block,
    topk_idx.expand(hkv, batch, topk).contiguous(),
)
```

## Scope

- Target family: NVIDIA Blackwell CUDA compute capability 12.x.
- Builder target: CUDA 12.8+ with `cuda-capabilities = ["12.0", "12.1"]`.
- Published artifact metadata: CUDA 12.8 variants currently list `12.0`;
  CUDA 13.0/13.2 variants list both `12.0` and `12.1`.
- Validated hardware: DGX Spark / GB10 / SM121.
- Validated MiniMax shape: query heads `64`, KV heads `4`, head dim `128`,
  sparse block `128`, top-k blocks `16`.
- Validated context lengths: `128`, `2048`, `4096`, `32768`.
- Correctness gate: cosine similarity `>= 0.999` against paged FP32 PyTorch
  references; native score-to-top-k selection matches PyTorch blockmax top-k
  set semantics.

Implementation note: this package includes a native CUDA top-k helper and a
Blackwell-validated Triton CUDA decode sparse attention fallback. A full native
CUTE/CUDA attention body aligned with upstream `MiniMaxAI/msa` is the next
implementation step.

Source provenance and validation details are documented in `SYNC.md` and
`VALIDATION.md`.
