# FP8 Prefill Attention Blackwell

FlashRT native CUDA FP8 causal GQA self-attention for Blackwell. The v1 API
deliberately exposes only the production-validated contract; unsupported
layouts fail explicitly rather than silently selecting an unverified path.

```python
from kernels import get_kernel

k = get_kernel("flashrt/fp8-prefill-attention-blackwell", version=1)
out = k.fp8_causal_gqa_attention_bf16(q, key, value)
```

Inputs use contiguous NHD layout: `q [S,32,128]`, `key/value [S,8,128]`,
FP8 E4M3FN. `S` must be a multiple of 128 and at least 256. Output is BF16 and the
operation is causal. CUDA 12.8+ and Blackwell SM120/SM121 are required.

See [CARD.md](CARD.md) for the complete contract and [VALIDATION.md](VALIDATION.md)
for release gates. The complete FlashRT runtime and serving pipeline lives at
https://github.com/flashrt-project/FlashRT.
