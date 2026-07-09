# torch-ext

PyTorch bindings and Python package for `flashrt-flex-attention-train`.

`torch_binding.cpp` registers a package marker so HF `kernel-builder` can load
the extension. The Python module provides the SDPA fallback and the stable
training API used by PI052 integration.
