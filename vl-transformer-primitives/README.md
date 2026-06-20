# VL Transformer Primitives

FlashRT CUDA kernels for vision-language transformer hot-path helpers.

This package is intentionally named by use case, not by one model family. The
current APIs include Qwen3/Qwen3-VL decode post-processing contracts and a
generic BF16 vision-token pooling primitive used by Qwen3-VL/Cosmos3-style
understanding towers.

## Available Functions

- `qwen3_q_norm_rope_qstage_bf16`: Q RMSNorm + full RoPE + decode staging write.
- `qwen3_k_norm_rope_kvwrite_bf16`: K RMSNorm + full RoPE + K/V cache slot write.
- `qwen3_k_norm_rope_kvwrite_devpos_bf16`: device-position K/V cache write for CUDA Graph decode loops.
- `avg_pool_vision_tokens_bf16`: spatial average pooling for flattened BF16 vision tokens.

## Usage

```python
from kernels import get_kernel

vl = get_kernel("flashrt/vl-transformer-primitives")

q_out = vl.qwen3_q_norm_rope_qstage_bf16(q_pre, q_norm_weight, cos, sin)
k_slot, v_slot = vl.qwen3_k_norm_rope_kvwrite_bf16(
    k_pre, v_pre, k_norm_weight, cos, sin
)
pooled = vl.avg_pool_vision_tokens_bf16(tokens, nv=2, h=16, w=16, pool_factor=2)
```

All tensors are CUDA contiguous BF16 tensors. Unsupported shapes fail at the
Python/C++ boundary instead of silently falling back.

## Shape Contracts

Qwen3 decode post-processing:

- `q_pre`: `(n_q_heads, 128)` BF16
- `k_pre`, `v_pre`: `(n_kv_heads, 128)` BF16
- norm weights: `(128,)` BF16
- `cos`, `sin`: `(64,)` BF16
- device-position cache: `(max_seq_len, n_kv_heads, 128)` BF16 with an int32 scalar position tensor

Vision-token pooling:

- input: `(nv * h * w, dim)` BF16
- output: `(nv * h / pool_factor * w / pool_factor, dim)` BF16
- `h` and `w` must be divisible by `pool_factor`

## Validation

See `VALIDATION.md` and `benchmarks/RESULTS.md`.
