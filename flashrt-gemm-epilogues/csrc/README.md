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
a shape-aware policy.

The BF16 GEMM epilogue wrapper uses a 32 MiB cuBLASLt workspace by default.
Override it for local sweeps with:

```bash
FLASHRT_GEMM_WORKSPACE_MB=256 python internal-tests/flashrt-gemm-epilogues/local_benchmark.py --suite --only gemm
```

Valid override values are positive MiB values up to `1024`.
