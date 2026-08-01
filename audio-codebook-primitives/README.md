# Audio Codebook Primitives

Native CUDA delayed-codebook selection and embedding kernels used by the
Higgs Audio/TTS autoregressive decode pipeline. No unrelated audio-model
operators are included.

## Functions

- `delayed_codebook_argmax_embed_bf16`
- `delayed_codebook_sample_embed_bf16`

Both functions consume BF16 logits `(num_codebooks, codebook_vocab)` and a
BF16 codebook `(num_codebooks, codebook_vocab, hidden)`. They return INT64
codes and the BF16 sum of the selected codebook embeddings. Codebooks with
index greater than `delay` emit `boc`.

```python
from kernels import get_kernel

ops = get_kernel(
    "flashrt/audio-codebook-primitives", version=1,
    trust_remote_code=True,
)
codes, embedding = ops.delayed_codebook_argmax_embed_bf16(
    logits, codebook, delay=3, boc=1024
)
```

The sampling API uses a deterministic counter-based RNG keyed by `seed`,
`step`, and codebook index. It does not claim bitwise equivalence with
`torch.multinomial`.

The first release is built with CUDA 13 so one artifact matrix can include
Ampere, Ada, Hopper, and SM100/103/110/120/121 without advertising an
architecture that the selected compiler cannot emit. Unsupported runtime
variants fail resolution rather than silently falling back.
