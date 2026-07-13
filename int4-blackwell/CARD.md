---
tags:
  - kernel
  - cuda
  - blackwell
  - int4
  - experimental
---

# FlashRT INT4 Blackwell

Experimental native E0M3/INT4 tensor-core primitives for NVIDIA Blackwell
GPUs. SM120/SM121 use `OMMA.SF.16864`; SM100/SM103/SM110 use the distinct
`tcgen05` block-scaled tensor-core path. Both paths select the uniform signed
INT4 codebook `0..7, -0, -1..-7`.

```python
from kernels import get_kernel
import torch

int4 = get_kernel("flashrt/int4-blackwell", version=2)
print(int4.codebook_probe("ab"))
# tensor([ 0., 1., 2., 3., 4., 5., 6., 7., 0., -1., ..., -7.])

# Asynchronous register-resident MMA probe for CUDA-event benchmarking.
scratch = torch.empty((680, 256), device="cuda", dtype=torch.float32)
int4.mma_probe("ab", iterations=8192, blocks=680, launches=20, out=scratch)
```

Available functions:

- `codebook_probe(mode="ab", device=None) -> Tensor[16]`
- `mma_probe(mode="ab", iterations=8192, blocks=None, launches=1, device=None, out=None) -> Tensor`
- `tcgen05_int4_gemm_bf16(a_packed, sfa_physical, b_packed, sfb_physical) -> Tensor`

On SM120/SM121, modes are `e2m1`, `a` (INT4 A), `b` (INT4 B), and `ab`
(INT4 A and B). The tcgen05 codebook canary currently exposes `ab`.

## Scope and support

- Packaged targets: SM100a, SM103a, SM110a, SM120a, and SM121a. The SM12x
  paths use bundled architecture-specific cubins; tcgen05 is compiled into the
  extension for CUDA 13.0 variants.
- Runtime-validated targets: SM120/SM120a and SM110/SM110a. SM121 carries the same generated
  instruction encoding and must pass the exact runtime canary before a device
  result is reported. SM100 and SM103 are build targets but remain runtime
  candidates until tested on those GPUs.
- Extension variants: CUDA 12.8 through 13.0; the bundled native cubins were
  generated with CUDA 13.0 and therefore require a CUDA 13.0-capable driver.
- The E0M3 selector bits are undocumented. SM12x uses reproducibly patched
  cubins. The tcgen05 backend uses a package-local CUTLASS descriptor override;
  it does not alter CUTLASS for any other package.
- `mma_probe` is an SM12x instruction-throughput probe. The tcgen05 GEMM API is
  experimental, requires M/N/K multiples of 128, and accepts CUTLASS physical
  UE4M3 scale layouts with conservatively sized backing storage. It is not a
  drop-in replacement for `torch.mm`.

On RTX 5090 (driver 580.159.03, CUDA 13.0.88), all 16 code points match the
uniform `0..7, -0, -1..-7` codebook exactly, all 128 accumulators agree, and
the INT4 x INT4 probe reaches 2026.8 TFLOPS versus 2026.6 TFLOPS for the same
register-resident E2M1 x E2M1 probe.

On NVIDIA Thor (SM110, CUDA 13.0.48), all 16 native tcgen05 E0M3 values match
`0..7, -0, -1..-7` exactly. A constant 128 x 128 x 128 GEMM produces the exact
expected BF16 tile for every code point.

See `SYNC.md` for provenance and the exact binary-rewrite contract.

## Credit

The SM120 `OMMA.SF` element-format bits were first documented publicly by the
**Ling Team**, author **@im0qianqian** (`@千千`). FlashRT reproduces and
productizes that finding and is extending its hardware validation. Read the
[original article](https://zhuanlan.zhihu.com/p/2059376150565089368).
