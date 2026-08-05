# Source Sync

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
