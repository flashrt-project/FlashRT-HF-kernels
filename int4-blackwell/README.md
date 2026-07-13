# int4-blackwell

See [CARD.md](CARD.md) for the public API, support boundary, and benchmark.

Rebuild the packaged SM120 and SM121 cubins with `scripts/rebuild_cubins.sh`
from a CUDA 13.0 environment before changing `probe.cu` or the CUDA toolkit
revision.

SM100, SM103, and SM110 use a separate CUTLASS tcgen05 backend compiled by
`kernel-builder` with CUDA 13.0. The package-local descriptor override selects
the hardware-validated E0M3 format without modifying CUTLASS globally.

The SM120 `OMMA.SF` element-format selector was first documented publicly by
the **Ling Team**, author **@im0qianqian** (`@千千`). FlashRT's implementation
reproduces, validates, packages, and extends that work. See the
[original Chinese article](https://zhuanlan.zhihu.com/p/2059376150565089368).
