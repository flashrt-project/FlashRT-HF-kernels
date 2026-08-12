# Source Sync

- Package: `adaptive-layernorm-producers` (Adaptive/diT LayerNorm and AdaLN-modulation FP8 producers).
- Upstream FlashRT source: `../official/FlashRT`
- Upstream revision: pending confirmation; packaged source is maintained in the
  flashrt-project FlashRT-HF-kernels repository
  (https://github.com/flashrt-project/FlashRT-HF-kernels).

Copied source files:

- `csrc/ada_layer_norm_fp8.cu`
- `csrc/ada_layer_norm_fp8.cuh`
- `csrc/ada_layer_norm_fp8_ptok.cu`
- `csrc/adaln_modulation6.cu`
- `csrc/adaln_modulation6.cuh`
- `csrc/dit_layer_norm_fp8.cu`
- `csrc/dit_layer_norm_fp8.cuh`

Local packaging edits:

- Added Tensor-facing PyTorch custom ops in `torch-ext/torch_binding.cpp`.
- Added Python wrappers and fake registrations in `torch-ext/adaptive_layernorm_producers`.
- Kept public APIs Tensor-facing; no raw pointer or stream arguments.
- Includes rewritten to be package-local; serving-runtime dependencies removed.
- CUDA launchers kept graph-safe: no dynamic allocation inside hot kernels.

Architecture assumptions:

- CUDA 12.8+ / 13.0+ (CUDA 13.2 validated on NVIDIA Thor, sm_110a).
- NVIDIA Blackwell-family targets; Thor sm_110a validated on real hardware.

Runtime constraints:

- Inputs and outputs are `torch.Tensor`; shapes and dtypes are validated in the binding.
- Benchmarks cap CUDA memory at 30 GB per process via `set_per_process_memory_fraction`.

Additional FP4 producer provenance:

- Upstream: `flashrt-project/FlashRT`
- GROOT N1.7 sync commit:
  `24df793f4fa2d50780aea03b644208c6e0cb4162`
- Copied sources: `csrc/dit_norm_fp4_sfa.cu/.cuh`

The package adds `sm110_fp4_dispatch.cu/.cuh` and Tensor-facing bindings around
the native producer. The SM110 implementation is an independent CUDA 13,
CUTLASS 4.4 target. Existing SM120 FP8/FP4 paths are unchanged.

The SM110 FP4 scale contract is E4M3 round-to-nearest. It must be checked
against the staged native BF16 norm plus native FP4 quantizer, including exact
packed values and swizzled scale bytes. Do not reuse the older SM120 ceil-scale
reference for this target.

`tests/native_reference/dit_bf16.cu` is an exact, test-only copy from the same
FlashRT revision. It keeps source and installed-artifact gates self-contained;
it is not listed in `build.toml` and is never linked into release artifacts.
