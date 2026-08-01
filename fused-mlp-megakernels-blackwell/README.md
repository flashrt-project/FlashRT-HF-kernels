# FlashRT Fused MLP Megakernels for Blackwell

This package exposes FlashRT's production FP16 GeGLU megakernel as a generic
PyTorch tensor API. It fuses the gate GEMM, GELU-tanh epilogue, up GEMM, and
elementwise product into one CUDA kernel and writes only the final hidden
activation.

```python
from kernels import get_kernel

ops = get_kernel("flashrt/fused-mlp-megakernels-blackwell", version=1)
hidden = ops.fp16_geglu_fused(x, gate_weight, up_weight)
```

The tensors are row-major FP16: `x [M,K]`, both weights `[N,K]`, and output
`[M,N]`. For an allocation-free CUDA Graph hot path, pass preallocated
`gate_scratch` and `output` buffers.

This package targets Blackwell SM100, SM103, and SM110 artifacts. It does not
claim SM120: the upstream kernel relies on the SM100-family tcgen05/TMA 2SM
instruction path and its production tile requires 213,504 bytes of shared
memory, while SM120 has a different execution/resource contract. A
specific architecture is claimed only after that artifact is built and tested;
unsupported devices fail explicitly. The implementation is synchronized from
the [FlashRT runtime](https://github.com/flashrt-project/FlashRT).
