# masked-mha-runtime

FlashRT native SM110/SM120 masked FP16/BF16 MHA for fixed-shape CUDA Graph runtimes.
It masks padded logits during softmax, supports fused-QKV token strides, and
keeps caller-owned logits/output buffers stable across graph replay.

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/masked-mha-runtime", version=1)
logits = ops.allocate_workspace(q, k)
out = torch.empty_like(q, memory_format=torch.contiguous_format)
ops.attention_mha_fp16_masked(q, k, v, logits=logits, out=out)
```

Public functions are `allocate_workspace`, `attention_mha_fp16_masked`,
`attention_mha_bf16_masked`, `forward_static`, `forward`, and
`forward_seqused_static`. The explicit masked names are additive; the legacy
`forward_static` entry dispatches the same kernels.
Inputs use `(sequence, heads, head_dim)`. The production GROOT gate covers
DiT `(41, 32, 48)`, ViT/LLM sequence lengths `277/1024`, padded boundaries
`1025/2048`, FP16 and BF16, fused strides, and bitwise CUDA Graph replay.
`forward_seqused_static` is the FP16 shared-KV path used by static VLA
decoders: Q is `(Sq,H,D)`, K/V are `(Sk_max,D)`, and a CUDA int32 `valid_k`
limits softmax without a standalone `-inf` fill. It supports `Sk_max <= 1024`.

The BF16 masked entry accepts `qkv_token_stride=` and validates it against the
actual Tensor token stride. This permits direct views into fused QKV GEMM
output while rejecting a mismatched layout before launch. Both masked entries
only read and write the valid `S_kv` columns, so padded workspace bytes never
participate in softmax or CUDA Graph replay.

See [CARD.md](CARD.md) for the complete contract. Source provenance is FlashRT
commit `24df793f4fa2d50780aea03b644208c6e0cb4162`.
