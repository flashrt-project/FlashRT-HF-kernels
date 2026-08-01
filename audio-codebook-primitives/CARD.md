---
library_name: kernels
license: apache-2.0
tags:
  - cuda
  - native-cuda
  - flashrt
  - audio
  - text-to-speech
---

# audio-codebook-primitives

FlashRT delayed-codebook argmax/sampling plus embedding-sum kernels for
autoregressive audio generation. See `README.md` for signatures and usage.

## Available functions

```python
delayed_codebook_argmax_embed_bf16(
    logits, codebook, *, delay, boc, codes=None, embedding=None
) -> (codes, embedding)

delayed_codebook_sample_embed_bf16(
    logits, codebook, *, delay, boc, temperature, seed, step,
    codes=None, embedding=None
) -> (codes, embedding)
```

`logits` is BF16 `[C,V]`, `codebook` is BF16 `[C,V,H]`, `codes` is INT64
`[C]`, and `embedding` is BF16 `[H]`. Supplying output buffers makes the call
allocation-free and CUDA Graph friendly. Sampling is deterministic for the
same `(seed, step)` but does not claim `torch.multinomial` bitwise parity.
