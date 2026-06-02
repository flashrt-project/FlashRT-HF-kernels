# Source Directory

Selected first source slices:

```text
official/FlashRT/csrc/kernels/fp4_w4a4_matvec_sm120.cu
official/FlashRT/csrc/kernels/fp4_w4a4_matvec_sm120.cuh
official/FlashRT/csrc/kernels/fp4_w4a4_mma_warpsplit_mrows_sm120.cu
official/FlashRT/csrc/kernels/fp4_w4a4_mma_warpsplit_mrows_sm120.cuh
official/FlashRT/csrc/kernels/megakernel/tinyfp8_kernels_sm120.cu
official/FlashRT/csrc/kernels/megakernel/tinyfp8_kernels_sm120.cuh
```

Keep source imports narrow; this package should not pull model-block
megakernels until their public Tensor API and shape story are ready.
