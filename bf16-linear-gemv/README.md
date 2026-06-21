# bf16-linear-gemv

FlashRT native CUDA BF16 M=1 decode GEMV kernels for transformer decode
projection paths.

## Functions

- `bf16_decode_gemv_bf16(x, weight, alpha=1.0, variant=0, out=None)`
- `bf16_decode_gemv_unrolled_bf16(x, weight, out=None)`

`x` is `(K,)` or `(1,K)` BF16, `weight` is `(N,K)` BF16, and output is
`(N,)` BF16. Unsupported shapes fail at the wrapper boundary.

## Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/bf16-linear-gemv", version=1, trust_remote_code=True)
x = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)
w = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16)
y = ops.bf16_decode_gemv_unrolled_bf16(x, w)
```

## Validation

```bash
python bf16-linear-gemv/tests/test_bf16_linear_gemv.py --backend source --mode full
```
