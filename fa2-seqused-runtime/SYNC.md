# Source synchronization

## Upstream revisions

- FlashRT repository: `https://github.com/LiangSu8899/FlashRT`
- FlashRT source revision: `b3eab55` (`refactor/pi05-native-frontend-pipeline`)
- Native C runtime origin: `bc48543` (`feat/fa2-native-c-runtime`)
- Vendored FlashAttention-2 kernel files retain their BSD-3-Clause license.

## Copied source

- `csrc/attention/fa2_wrapper.{cu,h}`
- `csrc/attention/fa2_wrapper_causal.cu`
- forward and split-KV instantiations for head dimensions 64/96/128/256
- BF16 causal instantiations for head dimensions 128/256
- required FlashAttention-2 forward headers

## Package-local changes

- Rewritten include paths to be package-local.
- Replaced raw pointer/stream Python bindings with validated Tensor operators.
- Added explicit unsupported-contract errors before native dispatch.
- Added explicit vector-alignment checks for padded BSHD layouts.
- Added setup helpers for exact split-KV workspace allocation.
- Added fake registrations for `torch.compile` tracing.
- Kept all allocations outside native hot-path operators.

## Build assumptions

- CUDA 12.8 or newer.
- SM80-family FA2 forward kernels; package variants target supported SM80+
  architectures enumerated in `build.toml`.
- CUTLASS is supplied by `kernel-builder` through `cutlass_3_6`.
- SM110/Thor is not part of this package's claimed support matrix because the
  upstream FlashRT FA2 runtime currently excludes that target.
