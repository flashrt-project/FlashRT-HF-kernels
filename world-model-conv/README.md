# FlashRT World Model Conv

Native CUDA kernels for world-model / video-diffusion convolution hot paths.

The first exported kernel is `fp8_conv3d_v18_ncdhw_res_bf16out`, a Blackwell
SM120a FP8 3D causal convolution with:

- virtual cache/new concat on the time axis,
- direct causal output over `T_new`,
- spatial 3x3 padding,
- fused per-channel bias,
- optional residual add,
- BF16 NCDHW output.

## Function

```python
from kernels import get_kernel

wmc = get_kernel("flashrt/world-model-conv")
out = wmc.fp8_conv3d_v18_ncdhw_res_bf16out(
    cache_x_fp8,  # (N, 2, H, W, Ci), float8_e4m3fn
    new_x_fp8,    # (N, T, H, W, Ci), float8_e4m3fn
    weight_fp8,   # (Co, 3, 3, 3, Ci), float8_e4m3fn
    bias_bf16,    # (Co,), bfloat16
    residual,     # (N, Co, T, H, W), bfloat16
    alpha=0.75,
)
```

For CUDA Graph/static-buffer runtimes:

```python
wmc.fp8_conv3d_v18_ncdhw_res_bf16out(
    cache_x_fp8, new_x_fp8, weight_fp8, bias_bf16, residual, alpha=0.75, out=out
)
```

## Shape Contract

- GPU target: Blackwell architecture-specific SM120a.
- `T_cache == 2`.
- `Ci % 32 == 0`.
- `Co % 8 == 0`.
- Input layout: NDHWC.
- Output/residual layout: NCDHW.

See `VALIDATION.md` and `benchmarks/RESULTS.md`.
