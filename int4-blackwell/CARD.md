---
tags:
  - kernel
  - cuda
  - blackwell
  - int4
  - experimental
---

# FlashRT INT4 Blackwell

Experimental native E0M3/INT4 tensor-core primitives for NVIDIA SM120 and
SM121 Blackwell GPUs. This package demonstrates that the SM12x
`OMMA.SF.16864` datapath executes a uniform sign-magnitude INT4 codebook at the
same issue rate as NVFP4 E2M1 on the validated SM120 implementation.

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

- Packaged targets: separate SM120a and SM121a cubins selected at runtime.
- Runtime-validated target: SM120/SM120a. SM121 carries the same generated
  instruction encoding and must pass the exact runtime canary before a device
  result is reported.
- Extension variants: CUDA 12.8 through 13.0; the bundled native cubins were
  generated with CUDA 13.0 and therefore require a CUDA 13.0-capable driver.
- The E0M3 selector bits are undocumented. The package ships reproducibly
  patched cubins and rejects other compute capabilities at runtime. SM100,
  SM103, and SM110 use the distinct `tcgen05`/`UTCOMMA` path and are not
  represented by these SM12x cubins.
- This is an instruction/codebook and issue-rate primitive, not a general
  matrix multiplication API. It must not be substituted for `torch.mm`.

On RTX 5090 (driver 580.159.03, CUDA 13.0.88), all 16 code points match the
uniform `0..7, -0, -1..-7` codebook exactly, all 128 accumulators agree, and
the INT4 x INT4 probe reaches 2026.8 TFLOPS versus 2026.6 TFLOPS for the same
register-resident E2M1 x E2M1 probe.

See `SYNC.md` for provenance and the exact binary-rewrite contract.

## Credit

The SM120 `OMMA.SF` element-format bits were first documented publicly by the
**Ling Team**, author **@im0qianqian** (`@千千`). FlashRT reproduces and
productizes that finding and is extending its hardware validation. Read the
[original article](https://zhuanlan.zhihu.com/p/2059376150565089368).
