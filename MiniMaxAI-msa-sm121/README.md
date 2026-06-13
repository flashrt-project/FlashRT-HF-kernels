# MiniMaxAI MSA SM121

`MiniMaxAI-msa-sm121` packages a standalone Python/Triton implementation of the
MiniMax M3 sparse attention decode path for SM121 / consumer Blackwell GPUs.

The original MiniMaxAI MSA Hub package is:

- <https://huggingface.co/kernels/MiniMaxAI/msa>

That package targets SM100 with CuTe-DSL kernels. This package is a hardware
extension path for SM121, keeping the MiniMax MSA semantics while using
architecture-portable Triton kernels validated by FlashRT on GB10 / SM121.

FlashRT also validated this decode-sparse path inside the MiniMax-Spark model
runtime on DGX Spark / GB10. The Hub package keeps only the standalone Tensor
API needed by the community; the full model runtime remains in FlashRT.

## Scope

Public APIs are exported from `minimaxai_msa_sm121`:

- `flash_decode_with_topk_idx`
- `flash_decode_with_gqa_share_sparse`

Primary release focus:

- decode sparse attention for MiniMax M3 shapes
- paged KV cache layout
- BF16 inputs
- context lengths 128 to 32768
- SM121 validation
- FlashRT MiniMax-Spark model-path validation on DGX Spark / GB10

The package does not require SGLang, vLLM, or FlashRT at runtime. It requires
PyTorch, Triton, CUDA, and a compatible NVIDIA GPU.

## Usage

```python
from kernels import get_kernel

msa = get_kernel(
    "flashrt/MiniMaxAI-msa-sm121",
    version=1,
    trust_remote_code=True,
)

out = msa.flash_decode_with_gqa_share_sparse(
    q,
    sink,
    k_cache,
    v_cache,
    req_to_token,
    seq_lens,
    slot_ids,
    block_size,
    topk_idx,
)
```

For local source testing:

```bash
PYTHONPATH=MiniMaxAI-msa-sm121/torch-ext \
  python MiniMaxAI-msa-sm121/tests/test_msa_sm121.py --quick
```

Full validation includes ctx32768:

```bash
PYTHONPATH=MiniMaxAI-msa-sm121/torch-ext \
  python MiniMaxAI-msa-sm121/tests/test_msa_sm121.py
```

## Attribution

This package vendors and standalone-adapts the MiniMax-M3 block-sparse attention
Triton kernels from SGLang PR #27944, preserving upstream copyright headers.
FlashRT provides the SM121 packaging, validation, and model-path integration.

See [SYNC.md](SYNC.md) for provenance and interface details.
