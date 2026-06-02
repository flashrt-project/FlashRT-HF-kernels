# Source Directory

Synced FlashRT CUDA/C++ sources for GEMM epilogues live here.

Do not include serving runtime files unless they are refactored into
package-local, Tensor-oriented helpers.

The FP8 quantize kernels use package-local row/column tile launchers. Override
the runtime block size for local sweeps with:

```bash
FLASHRT_QUANT_BLOCK_SIZE=512 python internal-tests/flashrt-gemm-epilogues/local_benchmark.py --suite --only quant
```

Valid override values are `128`, `256`, `512`, and `1024`. The default path uses
a shape-aware policy tuned on the first SM120 sweep:

- `bias_gelu_quantize_fp8_static_bf16`: 512 for bias decode micro-batches,
  1024 for M<=32 with N<=8192, 512 for biased VLA-width M<=32, otherwise 256.
- `gelu_quantize_fp8_static_bf16`: 256 for M=1 and VLA-width cases, 1024 for
  2<=M<=32 with N<=8192, otherwise 256.
- `channel_scale_quantize_fp8_static_bf16`: 1024 for M<=32 with N<=8192, 512
  for 4096-wide M<=128, otherwise 256.

The BF16 GEMM epilogue wrapper uses a 32 MiB cuBLASLt workspace by default.
Override it for local sweeps with:

```bash
FLASHRT_GEMM_WORKSPACE_MB=256 python internal-tests/flashrt-gemm-epilogues/local_benchmark.py --suite --only gemm
```

Valid override values are positive MiB values up to `1024`.
