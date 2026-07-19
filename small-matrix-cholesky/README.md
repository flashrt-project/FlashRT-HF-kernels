# Small-matrix Cholesky

`small-matrix-cholesky` provides a batched CUDA FP32 Cholesky factorization for
matrix orders 32, 64, and 128. It accepts arbitrary leading batch dimensions,
returns a dense lower-triangular factor, and explicitly zeros the upper
triangle.

The package is intentionally narrow: it is optimized for many independent
small symmetric positive-definite matrices and does not replace cuSOLVER for
general matrix sizes.

## Hub usage

```python
import torch
from kernels import get_kernel

ops = get_kernel(
    "flashrt/small-matrix-cholesky",
    version=1,
    trust_remote_code=True,
)

x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
a = x @ x.T + 0.5 * torch.eye(64, device="cuda")
l = ops.cholesky_small_fp32(a)
torch.testing.assert_close(l @ l.T, a, rtol=2e-4, atol=2e-4)
```

An optional preallocated output avoids allocation in repeated calls:

```python
out = torch.empty_like(a)
ops.cholesky_small_fp32(a, out=out)
```

## Contract

- CUDA, contiguous, `torch.float32` input and output;
- square last two dimensions with `n` in `{32, 64, 128}`;
- all leading dimensions are flattened into a batch;
- input matrices must be symmetric positive definite;
- input and output must not alias;
- output has the same shape as the input and a zero upper triangle.

The `n=128` path requests 66,048 bytes of dynamic shared memory per block.
Unsupported devices fail during launch rather than silently selecting a slower
implementation.

## Development

Run source correctness:

```bash
python tests/test_small_matrix_cholesky.py \
  --backend source --mode full \
  --registration-include /path/to/kernel-builder/templates/torch
```

Run the benchmark:

```bash
python benchmarks/benchmark.py \
  --backend source \
  --registration-include /path/to/kernel-builder/templates/torch
```

See `SYNC.md` for provenance and `VALIDATION.md` for the current promotion
status.
