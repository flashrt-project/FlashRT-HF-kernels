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

These are source-gate numbers, not Hub artifact claims. The same matrix must be
rerun from a cold-loaded `torch213-cxx11-cu132-x86_64-linux` artifact before
that variant is promoted.
