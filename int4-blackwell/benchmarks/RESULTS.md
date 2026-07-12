# RTX 5090 results

Environment: NVIDIA GeForce RTX 5090 (SM120), driver 580.159.03, CUDA 13.0.88.
The benchmark is a register-resident MMA issue-rate probe with 680 blocks,
eight warps per block, four independent accumulator chains, and 8192 loop
iterations. It excludes quantization, global-memory movement, and epilogues.

| Operand format | Throughput |
|---|---:|
| E2M1 x E2M1 | 2026.6 TFLOPS |
| INT4 x INT4 | 2026.8 TFLOPS |
| INT4 x E2M1 | 2027.9 TFLOPS |

These values establish equal native instruction throughput. They are not a
GEMM speedup claim and should not be compared directly with `torch.mm`.
