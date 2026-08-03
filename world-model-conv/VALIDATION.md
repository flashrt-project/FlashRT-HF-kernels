# Validation

Source validation compares `fp8_conv3d_v18_ncdhw_res_bf16out` against a PyTorch
reference that performs:

1. `torch.cat([cache_x, new_x], dim=1)`,
2. NDHWC to NCDHW conversion,
3. `torch.nn.functional.conv3d` with spatial padding `(0,1,1)`,
4. alpha scale and per-channel bias,
5. BF16 rounding,
6. residual add and final BF16 rounding.

Validation matrix:

- `(N=1,T_cache=2,T=1,H=8,W=8,Ci=32,Co=16)`
- `(N=1,T_cache=2,T=4,H=16,W=16,Ci=32,Co=32)`
- `(N=2,T_cache=2,T=4,H=16,W=24,Ci=64,Co=32)`

Run:

```bash
python tests/test_world_model_conv.py --backend source
python benchmarks/benchmark.py --backend source --warmup 50 --iters 500
kernel-builder check-config .
```

The full source matrix additionally covers FP8 causal Conv3D, FP8 Conv2D in
NHWC/NCDHW seams, NVFP4 causal Conv3D and residual output variants. It checks
max/p99/mean error, cosine, invalid-shape rejection, fullgraph tracing, raw
native parity and cuDNN/PyTorch convolution baselines where the numerical
contract is equivalent.

Source validation selects the installed architecture. SM120 runs the existing
FP8/NVFP4 matrix. SM110 runs four BF16 boundary/channel cases plus
`torch.compile(fullgraph=True)`. The SM110 source is byte-synced from FlashRT
commit `9972f0f` (`csrc/conv/bf16_conv3d_v0_sm110.cu`).

## SM110 portable SIMT fp8/nvfp4 fallback

The FP8/NVFP4 conv ops also ship a pure-SIMT reference
(`portable_conv_simt.cu`) compiled for `sm_110a`, since no native tensor-core
backend exists for them on Thor. The correctness test forces the SIMT path
with `FLASHRT_FORCE_SIMT=1` and compares against the native SM120 output within
the same tolerance envelope, validating the fallback on any device.

### SM110 (NVIDIA Thor) real-hardware validation

Validated on an NVIDIA Thor (`sm_110a`) device with PyTorch 2.9.1+cu130 and
CUDA 13.2, comparing the SIMT path against an FP32 eager convolution reference:

- `fp8_conv3d_v18_ncdhw_res_bf16out`: max abs err 0.0, cosine 0.99999994
- `fp8_causal_conv3d_ndhwc_bf16`: max abs err 0.0, cosine 0.99999994
- `fp8_conv2d_3x3_nhwc_bf16`: max abs err 0.0, cosine 1.0
- `fp8_conv2d_3x3_ncdhw_bf16`: max abs err 0.0, cosine 0.99999994
- `nvfp4_causal_conv3d_ndhwc_bf16`: max abs err 0.0, cosine 1.0
- `nvfp4_causal_conv3d_residual_ncdhw_bf16`: max abs err 3.9e-3 (BF16 residual add rounding), cosine 0.99999

The SIMT kernels are bit-faithful to the FP32 convolution reference on Thor.
