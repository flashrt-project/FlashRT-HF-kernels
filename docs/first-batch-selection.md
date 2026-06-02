# First Batch Selection

This repository should not publish every FlashRT kernel at once. The first
batch should contain kernels that are technically strong, fill a visible
ecosystem gap, and are easy for Hugging Face users to call from Tensor APIs.

## Ready Packages

| Package | First public surface | Why it leads |
| --- | --- | --- |
| `flashrt-vla-video` | Q/K/QKV split + RMSNorm + RoPE kernels | 20x+ launch-fusion wins on VLA, vision, video, and diffusion-style token post-processing. |
| `flashrt-gemm-epilogues` | FP8 quant epilogues and selected BF16 GEMM epilogues | Simple package format proof with clean PyTorch baselines and user-friendly APIs. |
| `flashrt-nvfp4` | NVFP4 scale-factor layout helpers | Small, reusable Blackwell low-bit building block that makes fused GEMM outputs inspectable. |

## Draft Packages To Populate Next

| Package | Selected first slice | Public reason |
| --- | --- | --- |
| `flashrt-smallm-gemm` | NVFP4 W4A4 decode matvec, W4A4 small-M warpsplit MMA, tiny FP8 small-M kernels | Decode and small-batch serving are latency dominated and poorly served by generic GEMM paths. |
| `flashrt-fused-quant` | SiLU/GELU activation + FP4/NVFP4 quant, residual RMSNorm + FP4/NVFP4 quant | These are memory-bound chains that users currently write as several PyTorch launches. |
| `flashrt-nvfp4` | W4A16 GEMM bias+GELU epilogues and Stream-K down GEMM | Follow-up slice after CUTLASS dependency isolation. |

## Promotion Rule

A draft package graduates only when it has:

- Tensor-only Python APIs with no FlashRT pointer ABI exposure.
- Correctness tests against PyTorch or a precise fake-quant reference.
- Benchmarks across generic shape grids plus at least one FlashRT-real shape
  family.
- Explicit CUDA architecture claims.
- Source provenance in `SYNC.md`.

The package can be a strong internal candidate before it is a public package;
the public repo should only advertise buildable surfaces.
