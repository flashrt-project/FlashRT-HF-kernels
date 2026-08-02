# FlashRT World Model Conv

Native CUDA kernels for world-model / video-diffusion convolution hot paths.

The package exports:

- `bf16_causal_conv3d_ndhwc_bf16` (experimental SM110 probe)
- `fp8_conv3d_v18_ncdhw_res_bf16out`
- `fp8_causal_conv3d_ndhwc_bf16`
- `fp8_conv2d_3x3_nhwc_bf16`
- `fp8_conv2d_3x3_ncdhw_bf16`
- `nvfp4_causal_conv3d_ndhwc_bf16`
- `nvfp4_causal_conv3d_residual_ncdhw_bf16`

The original `fp8_conv3d_v18_ncdhw_res_bf16out` is a Blackwell SM120a FP8
3D causal convolution with:

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

- GPU target: SM120a for FP8/NVFP4 functions; SM110a for the explicit BF16
  probe.
- `T_cache == 2`.
- `Ci % 32 == 0`.
- `Co % 8 == 0`.
- Input layout: NDHWC.
- Output/residual layout: NCDHW.

See `VALIDATION.md` and `benchmarks/RESULTS.md`.

The FP8 Conv2D paths cover NHWC and NCDHW tensor seams. The causal Conv3D
paths consume prepacked FP8 or NVFP4 activations/weights and expose optional
preallocated outputs for CUDA Graph runtimes. Unsupported channel, kernel,
stride and architecture combinations raise before launch.

The SM110 BF16 function is opt-in and is not the default backend. On the three
Cosmos3-Edge VAE sites it is accurate but slower than cuDNN; see the benchmark
ledger.
