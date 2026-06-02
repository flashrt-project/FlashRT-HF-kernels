# Benchmark Baselines

This document defines which benchmark baselines are acceptable for the v1
FlashRT HF kernel batch. PyTorch eager is useful for readability and HF runner
compatibility, but it is not always strong enough for headline claims.

## Baseline Classes

| Class | Meaning | Use |
| --- | --- | --- |
| PyTorch eager | Direct tensor ops in PyTorch with synchronization | Readable reference and HF benchmark runner compatibility |
| Unfused CUDA chain | Existing FlashRT or package-local CUDA launches without the fused kernel | Launch-count and fusion benefit |
| Vendor/library | cuBLASLt, CUTLASS, cuDNN, or other strong library path | Headline GEMM, convolution, and low-bit claims |
| Deterministic reference | Small exact or fake-quant reference | Correctness only, not performance headline |

## Package Policy

### `flashrt-gemm-epilogues`

- FP8 quant epilogues may use PyTorch eager as a public baseline because the
  operation is a memory-bound elementwise epilogue chain.
- BF16 GEMM epilogue headlines require a strong GEMM baseline in addition to
  `torch.addmm` or `gelu(torch.addmm)`.
- Weak GEMM shapes stay marked `watch` or `reject` even if they are useful
  compatibility coverage.

### `flashrt-vla-video`

- QKV split, RMSNorm, RoPE, and cache/stage helpers may use PyTorch eager as a
  public baseline because they replace many small launch-bound tensor ops.
- If a baseline is already available as separate FlashRT CUDA launches, include
  it as an additional internal comparison before making model-level claims.
- Report token/head grids, not only one model shape.

### `flashrt-nvfp4`

- The v1 layout helper is declared CUDA 12.8+ SM120-only.
- Layout-only helpers may use byte-parity correctness and PyTorch/CUDA tensor
  reshapes as readable baselines.
- Fused NVFP4 GEMM epilogues require CUTLASS/cuBLASLt or an unfused strong
  CUDA chain before becoming headline claims.
- SM120-only paths must be labeled CUDA 12.8+ SM120 until a non-SM120 source
  path is added.

### `flashrt-smallm-gemm`

- The v1 decode matvec source slice is declared CUDA 12.8+ SM120-only.
- PyTorch dequant plus matmul is acceptable as a readable baseline.
- Headline low-bit GEMM/GEMV claims require cuBLASLt/CUTLASS or a known strong
  FlashRT internal low-bit baseline when available.
- Dispatch boundaries must be benchmark-backed; do not claim a generic
  dispatcher until small-M and decode grids are swept.

### `flashrt-fused-quant`

- The v1 SwiGLU quantization source slice is declared CUDA 12.8+ SM120-only.
- PyTorch eager is acceptable as a readable baseline for
  `SiLU(gate) * up + quant`.
- Report effective memory bandwidth because this package is memory-bound.
- Residual/RMSNorm variants need aliasing correctness and separate unfused
  launch-chain baselines before headline claims.

## Required Metadata

Every public `RESULTS.md` table must state:

- GPU, driver, PyTorch version, CUDA runtime.
- Warmup count and measured iterations.
- Whether the timing path is HF benchmark runner, package-local source
  extension, FlashRT internal module, or built Hub artifact.
- Baseline class.
- Shape grid and tile policy.
- Hardware scope, especially for SM120-only kernels.
