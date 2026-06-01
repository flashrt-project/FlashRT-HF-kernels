# Torch Extension

Add `torch_binding.cpp` and `torch_binding.h` here when this package becomes
buildable.

Bindings must:

- Register ops with `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)`.
- Use `REGISTER_EXTENSION(TORCH_EXTENSION_NAME)`.
- Accept `torch::Tensor` objects and validate shape/dtype/device.
- Launch package-local CUDA functions.
