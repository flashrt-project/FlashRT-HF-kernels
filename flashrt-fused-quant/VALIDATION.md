# Validation: flashrt-fused-quant

Validated on June 2, 2026.

This package is still a draft package. The current validation record only
covers the first source slice:

- `silu_mul_quant_nvfp4_swizzled_bf16`
- `silu_mul_merged_quant_nvfp4_swizzled_bf16`

## Environment

Runtime smoke environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: 2.9.1+cu128
- CUDA capability: 12.0
- Build scope: CUDA 12.8+ and SM120

## Current Scope

The first draft wrappers fuse:

- `SiLU(gate) * up`
- BF16 round-trip semantics matching upstream FlashRT
- NVFP4 e2m1 packed output bytes
- CUTLASS Sm1xx swizzled UE4M3 scale-factor bytes

Supported by the synced source slice:

- split inputs: `gate, up` with shape `(rows, cols)`
- merged input: `merged_gate_up` with shape `(rows, 2 * cols)` and row layout
  `[gate | up]`
- BF16 contiguous CUDA inputs
- `cols` divisible by 16

## Source Accuracy Sweep

The local source-extension sweep compiled:

```text
torch-ext/torch_binding.cpp
csrc/silu_mul_to_nvfp4_swizzled.cu
```

The sweep uses the Hugging Face kernel-builder `registration.h` template
include path locally.

Command:

```bash
python scripts/accuracy_sweep.py --backend source --mode full --package flashrt-fused-quant
```

Result: passed 144 checks.

Correctness was checked against a Python fake-quant reference that reproduces:

- BF16 round-trip after SiLU;
- BF16 output of `SiLU(gate) * up`;
- UE4M3 ceil scale selection;
- FP4 e2m1 packing;
- CUTLASS Sm1xx scale-factor swizzle;
- zero-filled padding bytes in the swizzled scale buffer.

Covered shape grid:

- decode rows `1,2,4,8`, hidden `4096,8192,12288,16384`;
- small rows `16,32`, hidden `4096,8192,12288,16384`;
- prefill rows `64,128,256`, hidden `4096,8192,12288`;
- video rows `1024,2520`, hidden `4096,8192,12288`.

Both split and merged APIs passed packed-output and swizzled-scale byte parity
over the full grid.

## Known Gaps

- `build.toml`, `flake.nix`, and `flake.lock` are present.
- `/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker
  check-config .` passed for this package.
- `kernel-builder build --variant torch211-cxx11-cu128-x86_64-linux` passed
  for this package, and the copied artifact passed package tests, examples,
  installed accuracy sweep, and the local release-candidate benchmark runner.
- Full `kernel-builder build-and-copy` matrix has not been run for this
  package.
- Official Hub `kernels benchmark` has not been run after upload.
- Memory-bandwidth results are still pending.
- Runtime validation is currently RTX 5090 only; this v1 source slice is
  declared SM120-only in `build.toml`.
- Residual/RMSNorm and SFA variants are not yet exposed.
