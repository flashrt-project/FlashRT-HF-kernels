# SageAttention3 Blackwell Benchmark Results

## Cold-loaded CUDA 13 artifact

- Package: `flashrt/sageattention3-blackwell@v1`
- GPU: NVIDIA GeForce RTX 5090, SM120a
- Variant: `torch211-cxx11-cu130-x86_64-linux`
- Layout: contiguous NHD, `B=1`, `H=32`
- Timing: 10 warmup and 30 measured iterations with CUDA events
- All Sage3 buffers are allocated before timing and CUDA Graph capture
- Fused timing includes centering, padding, Q/K/V FP4 quantization, correction
  GEMM and FP4 attention
- SDPA and SageAttention2 use the same input tensors and output contract

| S | D | SDPA us | Sage2 us | Sage3 core+quant us | Sage3 fused eager us | Sage3 fused graph us | Graph vs SDPA | Fused/legacy cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6144 | 128 | 3100.482 | 1352.993 | 800.612 | 1019.613 | 995.471 | 3.11x | 1.00000000 |
| 24576 | 128 | 45867.896 | 18757.026 | 11516.395 | 13086.833 | 13110.972 | 3.50x | 1.00000000 |
| 2688 | 64 | 264.132 | N/A | 114.843 | 147.665 | 139.151 | 1.90x | 1.00000000 |

The installed artifact passed the full correctness matrix for D64/D128, both
block-mean modes, aligned and unaligned sequence lengths, explicit invalid
contracts, caller-owned pointer stability and bitwise CUDA Graph replay.

The strict long-video all-in target is `<= 13.000 ms`. The measured
`13.111 ms` is a **0.111 ms miss** and is not recorded as a pass. Profiling at
`S=24576,D=128` attributes about 1.28 ms to K reduction, Q/K/V quantization and
the BF16 correction GEMM; the attention core is about 11.6 ms.

## CUDA 12.8 compiler boundary

The cold-loaded `torch211-cxx11-cu128-x86_64-linux` diagnostic build produced
the following D128 numbers before the unsupported path was removed:

| S | D | SDPA us | Sage3 fused graph us | Graph vs SDPA | Decision |
|---:|---:|---:|---:|---:|---|
| 6144 | 128 | 3095.070 | 6334.589 | 0.49x | reject |
| 24576 | 128 | 45866.099 | 99722.393 | 0.46x | reject |

CUDA 12.8 generates roughly 896-1024 bytes of local stack per thread for the
upstream D128 template. Block-N and stage-count experiments did not remove the
spill, and exact upstream compiler flags reproduced it. The package therefore
advertises D64 only in CUDA 12.8 artifacts and fails fast for D128. CUDA 13.0+
artifacts advertise and run D64/D128. This is an artifact capability boundary,
not a correctness waiver.

The CUDA 12.8 D64 audio path remains supported and measured 143.629 us at
`S=2688,D=64` versus 262.333 us for SDPA (1.83x). Release validation requires a
cold-loaded clean artifact, D64 full correctness and graph replay, plus an
explicit D128 rejection test.
