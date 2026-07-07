# speculative-draft-primitives

Native CUDA speculative decoding helper kernels from FlashRT.

## Functions

- `argmax_bf16(logits, out=None)`
- `accept_greedy_bf16(logits, drafts, spec_k, argmax_out=None, accept_n=None)`
- `accept_partitioned_bf16(logits, drafts, spec_k, parts=None, argmax_out=None, accept_n=None, partial_vals=None, partial_idx=None)`

Use the partitioned variant for large-vocabulary drafter/verify loops. The
Python wrapper chooses a static partition count from the vocabulary size when
`parts` is omitted.

## Example

```python
from kernels import get_kernel
import torch

spec = get_kernel("flashrt/speculative-draft-primitives", version=1, trust_remote_code=True)

logits = torch.randn((16, 248320), device="cuda", dtype=torch.bfloat16)
drafts = torch.argmax(logits.float(), dim=1)[:15].contiguous()
argmax, accept_n = spec.accept_partitioned_bf16(logits, drafts, spec_k=15)
```

All inputs must be CUDA contiguous tensors. Unsupported shapes fail explicitly.
