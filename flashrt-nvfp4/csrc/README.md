# Source Directory

First source sync should be the NVFP4 scale-factor reshape helper from:

```text
official/FlashRT/csrc/quantize/nvfp4_sf_reshape_sm120.cu
official/FlashRT/csrc/quantize/nvfp4_sf_reshape_sm120.cuh
```

The fused GEMM epilogue sources come later because they pull CUTLASS and
architecture-specific build constraints into the package.
