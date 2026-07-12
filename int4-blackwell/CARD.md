---
tags:
  - kernel
  - cuda
  - blackwell
  - int4
  - experimental
---

# FlashRT INT4 Blackwell

Experimental native E0M3/INT4 tensor-core primitives for NVIDIA SM120
(GeForce RTX 50-series Blackwell). This package demonstrates that the SM120
`OMMA.SF.16864` datapath executes a uniform sign-magnitude INT4 codebook at the
same issue rate as NVFP4 E2M1.

```python
from kernels import get_kernel
import torch

int4 = get_kernel("flashrt/int4-blackwell", version=1)
print(int4.codebook_probe("ab"))
# tensor([ 0., 1., 2., 3., 4., 5., 6., 7., 0., -1., ..., -7.])

# Asynchronous register-resident MMA probe for CUDA-event benchmarking.
scratch = torch.empty((680, 256), device="cuda", dtype=torch.float32)
int4.mma_probe("ab", iterations=8192, blocks=680, launches=20, out=scratch)
```

Available functions:

- `codebook_probe(mode="ab", device=None) -> Tensor[16]`
- `mma_probe(mode="ab", iterations=8192, blocks=None, launches=1, device=None, out=None) -> Tensor`

Modes are `e2m1`, `a` (INT4 A), `b` (INT4 B), and `ab` (INT4 A and B).

## Scope and support

- Exact target: SM120/SM120a.
- Extension variants: CUDA 12.8 through 13.0; the bundled native cubins were
  generated with CUDA 13.0 and therefore require a CUDA 13.0-capable driver.
- The E0M3 selector bits are undocumented. The package ships reproducibly
  patched cubins and rejects other compute capabilities at runtime.
- This is an instruction/codebook and issue-rate primitive, not a general
  matrix multiplication API. It must not be substituted for `torch.mm`.

On RTX 5090 (driver 580.159.03, CUDA 13.0.88), all 16 code points match the
uniform `0..7, -0, -1..-7` codebook exactly, all 128 accumulators agree, and
the INT4 x INT4 probe reaches 2026.8 TFLOPS versus 2026.6 TFLOPS for the same
register-resident E2M1 x E2M1 probe.

See `SYNC.md` for provenance and the exact binary-rewrite contract.
