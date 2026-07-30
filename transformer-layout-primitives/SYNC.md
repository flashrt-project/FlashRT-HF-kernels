# Source Sync Notes

The CUDA kernels are adapted from FlashRT BF16 layout/text helper kernels and
re-exposed as Hugging Face Kernel Hub Tensor APIs.

Source provenance:

- FlashRT baseline: `official/FlashRT` commit `132049d`.
- Fused single-tensor RMSNorm+RoPE structure:
  `flash_wm/csrc/kernels_bf16.cu`.
- Q/K pair grid and GQA contract:
  `origin/cosmos-edge` commit `c2c7619`,
  `csrc/kernels/cosmos3_edge_misc.cu`.
- Package-local edits remove model names, raw pointers and explicit streams
  from the public API. The generic pair path supports even head dimensions
  from 8 through 256 and validates tensors before launch.

Public-package constraints:

- No raw pointer, stream, or FlashRT runtime-context arguments.
- All public functions accept and return PyTorch tensors.
- Shape contracts are checked in the C++ binding before launch.
- Keep model names out of the API surface.
