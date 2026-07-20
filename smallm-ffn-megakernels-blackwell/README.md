# Small-M FFN Megakernels Blackwell

Two full FlashRT FP8 FFN regions for fixed-shape, static-buffer inference:

- `fp8_gelu_ffn_gated_residual_bf16_static`: `D=1024`, hidden `4096`,
  `M<=32`; fuses up GEMM, bias, GELU, hidden requantization, down GEMM,
  bias, gate and residual.
- `fp8_gelu_ffn_residual_bf16_static`: `D=512`, hidden `2048`, `M<=144`
  (or `M<=192` with `split_stage=True`); also fuses BF16 input quantization.

```python
from kernels import get_kernel
k = get_kernel("flashrt/smallm-ffn-megakernels-blackwell", version=1)
y = k.fp8_gelu_ffn_gated_residual_bf16_static(...)
```

For CUDA Graph hot paths, allocate and reuse every `out`/scratch/barrier tensor.
See [CARD.md](CARD.md). Full model runtimes live at
https://github.com/flashrt-project/FlashRT.
