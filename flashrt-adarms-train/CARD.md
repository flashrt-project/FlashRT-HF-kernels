# flashrt-adarms-train

AdaRMS + gated residual BF16/FP32 training kernels.

## Available functions

- `adarms`
- `resgate_adarms`
- `adarms_forward`
- `resgate_adarms_forward`
- `FlashRTAdaRMSNorm`

## Acceptance status

- CUDA forward/backward kernels: available for supported CUDA BF16/FP32 shapes.
- Reference/eager autograd fallback: available for CPU, FP64 gradcheck, and
  unsupported shapes.
- Correctness Gate 1: passed.
- Performance Gate 2: 11/12; the remaining conditional case is large-row
  non-adaptive gated residual RMSNorm and is documented in
  `benchmarks/RESULTS.md`.
- Precision mode: BF16/FP32 training only, no FP8/FP4.

Recommended load path:

```python
from kernels import get_kernel
ops = get_kernel("flashrt/flashrt-adarms-train", revision="v1")
```
