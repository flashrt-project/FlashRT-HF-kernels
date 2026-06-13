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

Blackwell-family package for MiniMax MSA decode sparse attention, maintained by
FlashRT. The upstream package is
[`MiniMaxAI/msa`](https://huggingface.co/kernels/MiniMaxAI/msa), which targets
SM100. This package extends the decode-sparse path to NVIDIA Blackwell
compute capability 12.x and has been validated in the FlashRT MiniMax-Spark
runtime on DGX Spark / GB10 / SM121.

## Load

```python
from kernels import get_kernel

msa = get_kernel(
    "flashrt/MiniMaxAI-msa-blackwell",
    version=1,
    trust_remote_code=True,
)
```

## What You Can Call

### Official MiniMaxAI/msa names

| Function/class | Status in this Blackwell package |
|---|---|
| `sparse_decode_atten_func` | Available. Blackwell paged BF16/FP16 single-token decode wrapper. |
| `SparseDecodePagedAttentionWrapper` | Available. `plan(...).run(...)` wrapper for the same decode path. |
| `build_k2q_csr` | Available. CSR construction helper for the official prefill API. |
| `SparseK2qCsrBuilderSm100` | Available compatibility class; `build()` delegates to `build_k2q_csr`. |
| `Nvfp4QuantizedTensor` | Available metadata dataclass. |
| `quantize_bf16_to_nvfp4_128x4` | Available when Transformer Engine NVFP4 support is installed. |
| `quantize_kv_bf16_to_nvfp4_128x4` | Available when Transformer Engine NVFP4 support is installed. |
| `dequantize_nvfp4_128x4_to_bf16` | Available reference dequantizer. |
| `swizzle_nvfp4_scale_to_128x4` | Available scale-layout helper. |
| `nvfp4_global_scale_from_amax` | Available scale helper. |
| `sparse_atten_func` | Available. Official CSR sparse prefill API backed by the Blackwell Triton BF16/FP16 prefill kernel. |
| `sparse_atten_nvfp4_kv_func` | Available. Built artifacts use native CUDA swizzled NVFP4 -> BF16 dequantization, then call Blackwell sparse prefill. |
| `fp4_indexer_block_scores` | Available. Built artifacts use the native CUDA Blackwell block-score kernel and return the official `[Hq, ceil(max_seqlen_k/128), total_q]` score layout. |

### FlashRT Blackwell helper names

These are the direct low-level APIs used by the FlashRT MiniMax-Spark decode
path:

- `flash_decode_with_topk_idx`
- `flash_decode_with_gqa_share_sparse`
- `native_topk_from_scores`
- `native_nvfp4_dequant_swizzled_to_bf16`
- `has_native_ops`
- `naive_flash_decode_with_topk_idx`
- `naive_flash_decode_with_gqa_share_sparse`
- `get_cu_seqblocks`
- `robust_allocator`

## Decode Example

This example uses the official MiniMax decode-facing name
`sparse_decode_atten_func`.

```python
import torch
from kernels import get_kernel

msa = get_kernel("flashrt/MiniMaxAI-msa-blackwell", version=1, trust_remote_code=True)

B, Hq, Hkv, D = 1, 64, 4, 128
page_size = 128
num_pages = 32
topk = 16
device = "cuda"
dtype = torch.bfloat16

q = torch.randn(B, Hq, D, device=device, dtype=dtype)
k = torch.randn(num_pages, Hkv, page_size, D, device=device, dtype=dtype)
v = torch.randn_like(k)
page_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(B, -1)
seqused_k = torch.tensor([num_pages * page_size], device=device, dtype=torch.int32)
q2k_indices = torch.arange(topk, device=device, dtype=torch.int32).view(1, 1, topk)
q2k_indices = q2k_indices.expand(Hkv, B, topk).contiguous()

out = msa.sparse_decode_atten_func(
    q,
    k,
    v,
    q2k_indices,
    page_table=page_table,
    seqused_k=seqused_k,
    seqlen_q=1,
    max_seqlen_k=num_pages * page_size,
    blk_kv=page_size,
)
```

## Wrapper Example

```python
wrapper = msa.SparseDecodePagedAttentionWrapper(blk_kv=128)
wrapper.plan(
    page_table=page_table,
    seqused_k=seqused_k,
    seqlen_q=1,
    max_seqlen_k=num_pages * page_size,
    q2k_indices=q2k_indices,
    num_qo_heads=Hq,
    num_kv_heads=Hkv,
    head_dim=D,
)
out = wrapper.run(q, k, v)
```

## Prefill Example

This example uses the official MiniMax CSR prefill-facing name
`sparse_atten_func`.

```python
import torch
from kernels import get_kernel

msa = get_kernel("flashrt/MiniMaxAI-msa-blackwell", version=1, trust_remote_code=True)

T, Hq, Hkv, D = 512, 64, 4, 128
page_size = 128
topk = 16
device = "cuda"
dtype = torch.bfloat16

q = torch.randn(T, Hq, D, device=device, dtype=dtype)
k = torch.randn(T, Hkv, D, device=device, dtype=dtype)
v = torch.randn_like(k)
cu = torch.tensor([0, T], device=device, dtype=torch.int32)

q2k = torch.full((Hkv, T, topk), -1, device=device, dtype=torch.int32)
for qi in range(T):
    blocks = torch.arange(qi // page_size + 1, device=device, dtype=torch.int32)
    q2k[:, qi, : min(topk, blocks.numel())] = blocks[:topk]

k2q_row_ptr, k2q_q_indices = msa.build_k2q_csr(
    q2k, cu, cu, page_size, total_k=T
)
out = msa.sparse_atten_func(
    q,
    k,
    v,
    k2q_row_ptr,
    k2q_q_indices,
    topk,
    cu_seqlens_q=cu,
    cu_seqlens_k=cu,
    max_seqlen_q=T,
    max_seqlen_k=T,
    blk_kv=page_size,
)
```

## Direct FlashRT Decode Path

Use this lower-level path if you already have `topk_idx` in FlashRT's paged KV
layout.

```python
q = torch.randn(B, Hq, D, device=device, dtype=dtype)
k_cache = torch.randn(num_pages * page_size, Hkv, D, device=device, dtype=dtype)
v_cache = torch.randn_like(k_cache)
req_to_token = torch.arange(num_pages * page_size, device=device, dtype=torch.int32).view(B, -1)
seq_lens = torch.tensor([num_pages * page_size], device=device, dtype=torch.int32)
slot_ids = torch.zeros(B, device=device, dtype=torch.int64)
topk_idx = torch.arange(topk, device=device, dtype=torch.int32).view(1, 1, topk)
topk_idx = topk_idx.expand(Hkv, B, topk).contiguous()

out = msa.flash_decode_with_gqa_share_sparse(
    q, None, k_cache, v_cache, req_to_token, seq_lens, slot_ids, page_size, topk_idx
)
```

## Scope

- Target family: NVIDIA Blackwell CUDA compute capability 12.x.
- Builder target: CUDA 12.8+ with `cuda-capabilities = ["12.0", "12.1"]`.
- Validated hardware: DGX Spark / GB10 / SM121.
- Validated MiniMax shape: query heads `64`, KV heads `4`, head dim `128`,
  sparse block/page size `128`, top-k blocks `16`.
- End-to-end model validation: MiniMax-Spark runtime on GB10 through `32768`
  context length.
- Standalone kernel long-context validation: `128`, `2048`, `4096`, `32768`,
  `65536`, `131072`.
- Correctness gate: cosine similarity `>= 0.999` against paged FP32 PyTorch
  references; official decode wrapper output matches the direct Blackwell
  decode kernel.

## Implementation Notes

This package contains:

- native CUDA score-to-top-k helper;
- native CUDA tensor-core sparse decode route for the MiniMax-M3 Blackwell shape;
- native CUDA FP4 block-score indexer;
- native CUDA swizzled NVFP4 -> BF16 dequantization for the W4A16 quality path;
- Blackwell-validated sparse prefill attention wrapper;
- MiniMaxAI/msa-compatible Python API layer for decode, prefill, CSR, NVFP4,
  and FP4 block-score helpers.

When loaded from Hub built artifacts, the decode, FP4 indexer, and NVFP4
dequant hot paths use compiled CUDA ops. The source-tree mode keeps reference
paths so the API and correctness tests remain runnable before a wheel/shared
object has been built.

Source provenance and validation details are documented in `SYNC.md` and
`VALIDATION.md`.
