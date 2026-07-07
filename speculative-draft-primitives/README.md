# speculative-draft-primitives

FlashRT native CUDA helper kernels for speculative decoding control paths.

This package is intentionally model-neutral. It provides Tensor APIs for the
small but latency-sensitive operations around drafter/verify loops: per-row
argmax over BF16 logits and accepted-prefix computation against draft tokens.

## Available Functions

- `argmax_bf16(logits, out=None)`
- `accept_greedy_bf16(logits, drafts, spec_k, argmax_out=None, accept_n=None)`
- `accept_partitioned_bf16(logits, drafts, spec_k, parts=None, argmax_out=None, accept_n=None, partial_vals=None, partial_idx=None)`

## Tensor Contract

- `logits`: CUDA contiguous BF16 tensor, shape `(rows, vocab)`.
- `drafts`: CUDA contiguous int64 tensor, shape `(>= spec_k,)`.
- `argmax_out`: CUDA contiguous int64 tensor, shape `(rows,)`.
- `accept_n`: CUDA contiguous int32 tensor with at least one element.
- `partial_vals`: CUDA contiguous FP32 tensor, shape `(rows, parts)`.
- `partial_idx`: CUDA contiguous int32 tensor, shape `(rows, parts)`.

Unsupported shapes fail at the API boundary instead of silently falling back.

## Minimal Usage

```python
from kernels import get_kernel
import torch

spec = get_kernel("flashrt/speculative-draft-primitives", version=1, trust_remote_code=True)

logits = torch.randn((16, 32000), device="cuda", dtype=torch.bfloat16)
argmax = spec.argmax_bf16(logits)

drafts = argmax[:15].clone()
argmax, accept_n = spec.accept_greedy_bf16(logits, drafts, spec_k=15)
```

Use `accept_partitioned_bf16` for very large vocabularies where splitting the
row reduction into multiple column partitions improves occupancy. When `parts`
is omitted, the Python wrapper chooses a static default from the vocabulary
size: `8` for medium vocabularies, `16` above 65k, and `32` above 131k.

## Validation

```bash
python speculative-draft-primitives/tests/test_speculative_draft_primitives.py --backend source --mode full
python speculative-draft-primitives/benchmarks/benchmark.py --backend source --mode headline
```
