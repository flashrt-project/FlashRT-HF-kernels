# MiniMaxAI MSA Blackwell

`MiniMaxAI-msa-blackwell` is FlashRT's Blackwell-family extension package for
the MiniMax MSA decode-sparse path.

Upstream reference:

- MiniMaxAI MSA Hub package: <https://huggingface.co/kernels/MiniMaxAI/msa>
- MiniMaxAI MSA source: <https://github.com/MiniMax-AI/MSA>

The upstream Hub package targets SM100. This package targets Blackwell-family
CUDA compute capability 12.x GPUs and is validated in FlashRT's MiniMax-Spark
model path on DGX Spark / GB10 (`sm_121`). The current package is a native CUDA
package, but the native port is
not yet complete:

- native CUDA helper: score tensor -> top-k sparse block ids
- Triton CUDA fallback: Blackwell-validated decode score and sparse GQA attention

The next native alignment step is to port the upstream CUTE-DSL attention path
itself to Blackwell. Until that lands, this package is best described as a native
helper + Blackwell Triton attention package, not as a full native-CUTE replacement
for `MiniMaxAI/msa`.

## Load From Kernel Hub

```python
from kernels import get_kernel

msa = get_kernel(
    "flashrt/MiniMaxAI-msa-blackwell",
    version=1,
    trust_remote_code=True,
)
```

## Public APIs

The package exports:

- `native_topk_from_scores(score, seq_lens, block_size, topk)`: native CUDA
  helper that converts an already-computed `[heads, batch, max_blocks]` FP32
  score tensor to `[heads, batch, topk]` sparse block ids.
- `flash_decode_with_topk_idx(...)`: MiniMax decode indexer. With
  `disable_index_value=True`, returns `(None, topk_idx, real_seq_lens)`.
- `flash_decode_with_gqa_share_sparse(...)`: block-sparse GQA decode attention
  consuming `topk_idx`.
- `naive_flash_decode_with_topk_idx(...)` and
  `naive_flash_decode_with_gqa_share_sparse(...)`: PyTorch references for
  correctness checks.

MiniMax-M3 production decode shape:

| Field | Value |
|---|---:|
| Query heads | 64 |
| KV heads | 4 |
| Head dim | 128 |
| Sparse block | 128 |
| Top-k blocks | 16 |
| Input dtype | BF16 |

## Minimal Native Helper Example

This is the smallest native-CUDA call in the package. It is useful when another
runtime already computes MiniMax MSA block scores.

```python
import torch
from kernels import get_kernel

msa = get_kernel("flashrt/MiniMaxAI-msa-blackwell", version=1, trust_remote_code=True)

score = torch.randn(64, 1, 256, device="cuda", dtype=torch.float32)
seq_lens = torch.tensor([32768], device="cuda", dtype=torch.int32)
topk_idx = msa.native_topk_from_scores(score, seq_lens, block_size=128, topk=16)
```

## Minimal Decode Attention Example

```python
import torch
from kernels import get_kernel

msa = get_kernel("flashrt/MiniMaxAI-msa-blackwell", version=1, trust_remote_code=True)

batch, hq, hkv, d = 1, 64, 4, 128
ctx, block, topk = 4096, 128, 16

q = torch.randn(batch, hq, d, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(ctx, hkv, d, device="cuda", dtype=torch.bfloat16)
v_cache = torch.randn(ctx, hkv, d, device="cuda", dtype=torch.bfloat16)
q_index = torch.randn(batch, 1, d, device="cuda", dtype=torch.bfloat16)
k_index = torch.randn(ctx, 1, d, device="cuda", dtype=torch.bfloat16)
req_to_token = torch.arange(ctx, device="cuda", dtype=torch.int32).view(1, -1)
seq_lens = torch.tensor([ctx], device="cuda", dtype=torch.int32)
slot_ids = torch.zeros(batch, device="cuda", dtype=torch.int64)

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

The example intentionally uses a separate 1-head indexer Q/K path. MiniMax M3
does not select sparse blocks by running the 64 query heads directly as the
attention heads; it produces a compact sparse block set and broadcasts that
selection to the KV heads.

For a runnable version of the example above:

```bash
python MiniMaxAI-msa-blackwell/examples/decode_sparse.py
```

Use `--source-tree` when running the package before it is uploaded to the Hub:

```bash
PYTHONPATH=MiniMaxAI-msa-blackwell/torch-ext \
  python MiniMaxAI-msa-blackwell/examples/decode_sparse.py --source-tree
```

## Local Validation

Source-tree validation:

```bash
PYTHONPATH=MiniMaxAI-msa-blackwell/torch-ext \
  python MiniMaxAI-msa-blackwell/tests/test_msa_blackwell.py --quick
```

Full context validation:

```bash
PYTHONPATH=MiniMaxAI-msa-blackwell/torch-ext \
  python MiniMaxAI-msa-blackwell/tests/test_msa_blackwell.py
```

After HF Jobs publishes v1, validate the installed Hub artifact rather than the
source tree.

## Provenance

This package is a FlashRT community package for the MiniMax MSA Blackwell hardware
extension. It uses:

- MiniMaxAI/msa as the native package/API reference
- SGLang/vLLM MiniMax sparse attention Triton paths as the Blackwell fallback and
  correctness baseline
- FlashRT MiniMax-Spark runtime validation on DGX Spark / GB10

See `SYNC.md` and `VALIDATION.md` for details.
