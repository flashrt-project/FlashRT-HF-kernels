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
