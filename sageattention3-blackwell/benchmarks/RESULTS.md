# SageAttention3 Blackwell Benchmark Results

## RTX 5090 source release gate

- GPU: NVIDIA GeForce RTX 5090, SM120a
- Runtime: PyTorch 2.11.0 + CUDA 12.8
- Layout: contiguous NHD, B=1, H=32
- Sage2 and Sage3 include quantization and attention with preallocated buffers
- Allocation and Sage3 centering/delta preparation are outside timing
- PyTorch SDPA consumes the same centered BF16 tensors
- Timing: 3 warmup, 5 measured iterations; CUDA events

| S | D | SDPA us | Sage2 static us | Sage3 static us | vs SDPA | vs Sage2 | Sage3 cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6144 | 128 | 3064.224 | 1665.734 | 767.616 | 3.99x | 2.17x | 0.98160940 |
| 24576 | 128 | 45837.778 | 19367.699 | 11393.433 | 4.02x | 1.70x | 0.98185158 |
| 2688 | 64 | 293.350 | N/A | 125.766 | 2.33x | N/A | 0.98138434 |

These are source-gate numbers, not Hub artifact claims.

## Cold-loaded Hub artifact gate

- Artifact: `flashrt/sageattention3-blackwell@v1`
- Variant: `torch211-cxx11-cu128-x86_64-linux`
- Comparison Sage2: independently loaded from
  `flashrt/sageattention2-blackwell@v1`
- GPU: NVIDIA GeForce RTX 5090
- Runtime: PyTorch 2.11.0 + CUDA 12.8
- Timing: 10 warmup and 30 measured iterations; caller-owned workspaces
- Allocation and Sage3 centering/delta preparation are outside timing

| S | D | SDPA us | Hub Sage2 us | Hub Sage3 us | vs SDPA | vs Sage2 | Sage3 cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6144 | 128 | 3080.484 | 1695.228 | 838.791 | 3.67x | 2.02x | 0.98156697 |
| 24576 | 128 | 45865.987 | 19940.633 | 12155.592 | 3.77x | 1.64x | 0.98142821 |
| 2688 | 64 | 277.037 | N/A | 115.639 | 2.40x | N/A | 0.98154640 |

The same cold-loaded artifact passed the complete correctness matrix for
`S={128, 2688, 6144, 24576}`, `D={64,128}`, both block-mean modes, bitwise
CUDA Graph replay, and explicit invalid-contract rejection. Torch 2.12/CUDA
13.2 and Torch 2.13/CUDA 13.2 artifacts were built and published by the same
release job; their runtime qualification remains hardware/environment
specific and is not implied by the RTX 5090 row above.
