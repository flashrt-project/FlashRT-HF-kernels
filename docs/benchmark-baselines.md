# Benchmark Baselines

This document defines which benchmark baselines are acceptable for the v1
FlashRT HF kernel batch. PyTorch eager is useful for readability and HF runner
compatibility, but it is not always strong enough for headline claims.

Package-level comparison requirements and headline gates are defined in
`docs/kernel-comparison-matrix.md`. This file defines the baseline classes that
those package matrices refer to.

## Baseline Classes

| Class | Meaning | Use |
| --- | --- | --- |
| PyTorch eager | Direct tensor ops in PyTorch with synchronization | Readable reference and HF benchmark runner compatibility |
| `torch.compile` | Compiled PyTorch reference path, compile time excluded | Required general-purpose baseline only when the compiled reference is numerically equivalent to the eager reference |
| Unfused CUDA chain | Existing FlashRT or package-local CUDA launches without the fused kernel | Launch-count and fusion benefit |
| Vendor/library | cuBLASLt, CUTLASS, cuDNN, or other strong library path | Headline GEMM, convolution, and low-bit claims |
| Deterministic reference | Small exact or fake-quant reference | Correctness only, not performance headline |

## Package Policy

### Correctness Before Timing

- Correctness is always checked against the eager PyTorch or deterministic
  reference first.
- A `torch.compile` baseline is reportable only after the compiled reference
  output is verified against the eager reference for the same inputs and
  tolerance.
- If `torch.compile` changes a fake-quant or low-bit reference enough to cross
  quantization boundaries, mark the compiled baseline as unsupported instead of
  publishing a misleading speedup.
- For `flashrt-fp8-ffn`, the full
  `FP8 GEMM -> GELU -> FP8 requant -> FP8 GEMM` PyTorch reference is
  bit-exact under eager and `torch.compile(..., backend="aot_eager")`, but a
  raw default-Inductor compile of the whole fake-quant chain is not
  bit-equivalent on the current RTX 5090 validation stack. The package therefore
  reports a segmented compile-stable reference that graph-breaks the
  numerically sensitive FP8 requantization and final BF16 bias/cast boundaries,
  while still compiling the FP8 dequant GEMM regions. The runner verifies this
  compiled-reference output against eager output before timing it.

### `flashrt-gemm-epilogues`

- FP8 quant epilogues may use PyTorch eager as a public baseline because the
  operation is a memory-bound elementwise epilogue chain.
- Also report `torch.compile` for FP8 quant epilogues when supported by the
  local PyTorch version.
- BF16 GEMM epilogue headlines require a strong GEMM baseline in addition to
  `torch.addmm`, compiled `torch.addmm`, or `gelu(torch.addmm)`.
- Weak GEMM shapes stay marked `watch` or `reject` even if they are useful
  compatibility coverage.

### `flashrt-vla-video`

- QKV split, RMSNorm, RoPE, and cache/stage helpers may use PyTorch eager as a
  public baseline because they replace many small launch-bound tensor ops.
- `torch.compile` is the preferred public baseline for this package because it
  tests whether PyTorch's compiler can recover the same fusion opportunity.
- If a baseline is already available as separate FlashRT CUDA launches, include
  it as an additional internal comparison before making model-level claims.
- Report token/head grids, not only one model shape.

### `flashrt-nvfp4`

- The v1 layout helper is declared CUDA 12.8+ SM120-only.
- Layout-only helpers may use byte-parity correctness and PyTorch/CUDA tensor
  reshapes as readable baselines.
- Report `torch.compile` for tensor-layout references when compilation is
  supported. If the exact byte-layout reference uses Python loops or CPU copies,
  mark the compiled baseline as unsupported rather than forcing it.
- Fused NVFP4 GEMM epilogues require CUTLASS/cuBLASLt or an unfused strong
  CUDA chain before becoming headline claims.
- SM120-only paths must be labeled CUDA 12.8+ SM120 until a non-SM120 source
  path is added.

### `flashrt-smallm-gemm`

- The v1 decode matvec source slice is declared CUDA 12.8+ SM120-only.
- PyTorch dequant plus matmul is acceptable as a readable baseline.
- Also report compiled PyTorch dequant plus matmul when it is supported. Keep
  it separate from stronger low-bit library baselines.
- Headline low-bit GEMM/GEMV claims require cuBLASLt/CUTLASS or a known strong
  FlashRT internal low-bit baseline when available.
- Dispatch boundaries must be benchmark-backed; do not claim a generic
  dispatcher until small-M and decode grids are swept.

### `flashrt-fused-quant`

- The v1 SwiGLU quantization source slice is declared CUDA 12.8+ SM120-only.
- PyTorch eager is acceptable as a readable baseline for
  `SiLU(gate) * up + quant`.
- `torch.compile` should be reported for the PyTorch fusion chain when it can
  compile the fake-quant/layout reference.
- Report effective memory bandwidth because this package is memory-bound.
- Residual/RMSNorm variants need aliasing correctness and separate unfused
  launch-chain baselines before headline claims.

### `fp4-fused-ops`

- FP4 producer rows must report the dequantized FP4/SFA error envelope against
  the FP16 math reference. Nonzero error is expected because the output is
  quantized.
- v2 producer rows may compare against v1 where v1 supports the shape, but byte
  identity is not required; use dequantized value parity plus residual aliasing
  correctness.
- Rows with no meaningful fused public baseline should report latency and
  explain that the kernel is a pipeline-continuity primitive.

### `fp4-gemm`

- The correctness reference must dequantize the exact FP4/SFA/SFB tensors
  consumed by the kernel, then run PyTorch GEMM on those dequantized tensors.
- Report each CUTLASS schedule variant separately. Do not imply that the widen
  schedule is optimal for small shapes.
- CUTLASS/cuBLASLt or FlashRT internal low-bit comparison is required before
  making a broad low-bit GEMM headline claim.
- FP4out GEMM variants remain internal until every public shape passes
  `can_implement`, correctness, and benchmark gates.

## Required Metadata

Every public `RESULTS.md` table must state:

- GPU, driver, PyTorch version, CUDA runtime.
- Warmup count and measured iterations.
- Whether the timing path is HF benchmark runner, package-local source
  extension, FlashRT internal module, or built Hub artifact.
- Baseline class.
- Result label from `docs/kernel-comparison-matrix.md`.
- `torch.compile` status for every row: `ok`, `unsupported`, `failed`, or
  `no_reference`.
- Shape grid and tile policy.
- Hardware scope, especially for SM120-only kernels.
