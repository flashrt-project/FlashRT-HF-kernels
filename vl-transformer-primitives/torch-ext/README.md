# Torch Extension

Python package: `vl_transformer_primitives`

The extension exposes CUDA tensor APIs plus fake registrations for
`torch.compile` tracing. The public wrappers allocate outputs by default and
also accept preallocated output tensors for static-buffer decode loops.
