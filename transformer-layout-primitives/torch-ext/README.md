# Torch Extension

The extension registers Tensor APIs under the package namespace produced by
`kernel-builder`. The Python wrapper adds allocation helpers and fake
registrations for `torch.compile` tracing.
