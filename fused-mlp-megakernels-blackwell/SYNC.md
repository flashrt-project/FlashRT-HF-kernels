# Source Synchronization

The CUDA implementation and its vendored CUTLASS overrides are synchronized
from `official/FlashRT/csrc/gemm/mega/`:

- source revision `bfcaf610f23e9aac1c32ffd6ef344d0ba27cf19e`

- `flashrt_megakernel_geglu.cu`
- `flashrt_megakernel_geglu_kernel.hpp`
- `sm100_smem_aux_visitor.hpp`
- the four files under `cutlass/`

The public PyTorch binding is maintained in this package. The launcher also
populates CUTLASS `KernelHardwareInfo`; the upstream pointer wrapper left it at
zero, which makes standalone kernel-builder initialization fail. The upstream
WIP encoder fallback and the multi-launch `g8` bundle are intentionally excluded.
