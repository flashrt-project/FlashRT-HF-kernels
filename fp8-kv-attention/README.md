# FP8 KV Attention

FlashRT native CUDA XQA kernel for BF16-query attention over FP8 E4M3 paged
K/V cache.

This package exposes the clean production-style FP8-KV path used by FlashRT
Qwen3.6 experiments: K/V are quantized when written to cache, and attention
reads the FP8 cache directly. It does not re-quantize BF16 K/V inside the
attention call.

## Functions

- `xqa_bf16_fp8kv(q, k_cache, v_cache, page_table=None, seq_lens=None, mask=None, ...)`
- `causal_spec_mask(q_seq, device="cuda")`
- `default_page_table(num_pages, device="cuda")`
- `allocate_workspace(q_seq, device="cuda", scratch_mb=256)`

## v1 Shape Contract

The first public package intentionally exposes the validated fixed XQA shape:

- Q/O: BF16, `(q_seq, 24, 256)` or `(1, 1, q_seq, 24, 256)`
- K/V cache: FP8 E4M3, `(pages, 128, 4, 256)`
- page size: `128`
- Q heads / KV heads: `24 / 4`
- head dim: `256`
- supported q_seq: `1 <= q_seq <= 32`
- target: Blackwell CUDA, CUDA 12.8+

Unsupported shapes fail at the Python/C++ boundary instead of silently falling
back to a slower reference.

## Usage

```python
from kernels import get_kernel
import torch

attn = get_kernel("flashrt/fp8-kv-attention", trust_remote_code=True)

q = torch.randn(1, 24, 256, device="cuda", dtype=torch.bfloat16)
k_cache = torch.empty(8, 128, 4, 256, device="cuda", dtype=torch.float8_e4m3fn)
v_cache = torch.empty_like(k_cache)

out = attn.xqa_bf16_fp8kv(q, k_cache, v_cache)
```

For static-buffer runtimes, preallocate every output and workspace tensor:

```python
pages = k_cache.shape[0]
page_table = attn.default_page_table(pages, device=q.device)
seq_lens = torch.tensor([[pages * 128]], device=q.device, dtype=torch.int32)
mask = attn.causal_spec_mask(q.shape[0], device=q.device)
semaphores, scratch = attn.allocate_workspace(q_seq=q.shape[0], device=q.device)
out = torch.empty_like(q)

attn.xqa_bf16_fp8kv(
    q,
    k_cache,
    v_cache,
    page_table,
    seq_lens,
    mask,
    out=out,
    semaphores=semaphores,
    scratch=scratch,
)
```

## Provenance

The XQA implementation vendors the FlashInfer XQA subset already used in
FlashRT. See `csrc/attention/flashinfer_xqa_src/VENDOR.md` for upstream
source details and license information.

## Validation

See `VALIDATION.md` and `benchmarks/RESULTS.md`.
