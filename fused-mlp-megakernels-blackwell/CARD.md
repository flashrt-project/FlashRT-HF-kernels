---
tags:
- kernels
- cuda
- blackwell
- mlp
- geglu
library_name: kernels
---

# fused-mlp-megakernels-blackwell

## Available functions

- `fp16_geglu_fused`

It computes `GELU(x @ gate_weight.T, approximate="tanh") *
(x @ up_weight.T)` using one native CUDA megakernel. See `README.md` for the
tensor contract and static-buffer usage.

```python
fp16_geglu_fused(
    input, gate_weight, up_weight, *, gate_scratch=None, output=None
) -> output
```

All tensors are contiguous row-major FP16. Input is `[M,K]`, both weights are
`[N,K]`, and scratch/output are `[M,N]`. Pass both buffers for allocation-free
CUDA Graph replay. CUDA 13 and SM100/103/110 are required; SM120 is explicitly
outside this kernel's tcgen05/TMA 2SM contract.
