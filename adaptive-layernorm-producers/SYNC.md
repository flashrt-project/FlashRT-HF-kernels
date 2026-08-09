# Source Sync

- Package: `adaptive-layernorm-producers` (Adaptive/diT LayerNorm and AdaLN-modulation FP8 producers).
- Upstream FlashRT source: `../official/FlashRT`
- Upstream revision: `pending confirmation (upstream FlashRT revision not available in this checkout)`

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
