# Torch extension

The extension registers a Tensor-only out operator and exposes an allocating
Python helper. No raw pointers, caller-owned streams, or FlashRT runtime state
are part of the public API.
