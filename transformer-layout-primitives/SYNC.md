# Source Sync Notes

The CUDA kernels are adapted from FlashRT BF16 layout/text helper kernels and
re-exposed as Hugging Face Kernel Hub Tensor APIs.

Public-package constraints:

- No raw pointer, stream, or FlashRT runtime-context arguments.
- All public functions accept and return PyTorch tensors.
- Shape contracts are checked in the C++ binding before launch.
- Keep model names out of the API surface.
