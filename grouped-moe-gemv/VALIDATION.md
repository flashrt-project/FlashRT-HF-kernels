# Validation

Required source gate:

```bash
python grouped-moe-gemv/tests/test_grouped_moe_gemv.py --backend source --mode full
```

Expected source total for version 2: `22/22`.

The full gate covers:

- version 1 W4A16 API regression;
- W4A4 `K%16` contract, including non-MMA `K=80`;
- decode/verify gate-up layouts (`M=1/7, top_k=8`);
- decode/verify down layouts (`M=8/56, top_k=1`);
- exact target projections `(N,K)=(1024,2048)` and `(2048,512)`;
- bitwise grouped-vs-per-route native parity;
- max/p99/mean/cosine against the dequantized NVFP4 mathematical reference;
- separately reported NVFP4-vs-source-BF16 quality;
- allocation-free CUDA Graph replay, bitwise replay stability, and device
  `expert_idx` mutation between replays.

Correctness and low-precision quality are intentionally separate contracts.
The source-BF16 comparison is not used to hide or relabel implementation error.

## SM110 (Thor) portable SIMT fallback

The block-scaled mma path is SM120-only: the cute `SM120_16x8x64_TN_VS` atom
asserts at runtime on any arch without `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED`
(`cute/arch/mma_sm120.hpp`). On SM11x devices (and under
`FLASHRT_FORCE_SIMT=1`) `torch_binding.cpp` routes every W4A4 shape to the
portable SIMT reference kernel that ships in the same translation unit;
SM120 keeps the validated mma kernel unchanged.

Thor validation (NVIDIA Thor, SM110, CUDA 13.0, Torch 2.11, HF installed
artifact):

```bash
python grouped-moe-gemv/tests/test_grouped_moe_gemv.py \
  --backend installed --mode full
```

Result: `passed 22/22`. W4A4 contract cosine across the full grid is
`>= 0.999998` with the same NVFP4-vs-source-BF16 quality as the SM120 path
(e.g. `M=7,top_k=8,N=128,K=512` contract cosine `0.9999987`, p99 `0.0025`).
