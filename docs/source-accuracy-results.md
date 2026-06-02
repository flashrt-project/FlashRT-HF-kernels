# Source Accuracy Results

Validated on June 2, 2026 on NVIDIA GeForce RTX 5090, PyTorch 2.9.1+cu128,
CUDA capability 12.0.

These results use local source-extension builds through:

```bash
python scripts/accuracy_sweep.py --backend source --mode full --package all
```

This is the prebuild correctness gate. It is separate from built-artifact
tests, Hub benchmark runner results, and multi-hardware validation.

## Summary

| Package | Checks | Accuracy contract | Result |
| --- | ---: | --- | --- |
| `flashrt-gemm-epilogues` | 45 | FP8 output exact parity for bias+GELU, GELU, and channel-scale quant epilogues over the v1 shape grid | pass |
| `flashrt-vla-video` | 110 | BF16 Q/K outputs `max_abs <= 0.03125`, `max_rel <= 0.05` with `rel_floor=1`; V cache copy byte parity | pass |
| `flashrt-nvfp4` | 13 | NVFP4 scale-factor swizzle byte parity over the v1 layout grid | pass |
| `flashrt-smallm-gemm` | 12 | Constant and random/dequant references over `K in {4096,12288}` and `N in {1024,4096,12288}`; source sweep measured BF16 output `max_ulp <= 4` | pass |
| `flashrt-fused-quant` | 144 | Split and merged `SiLU(gate) * up -> NVFP4` packed bytes and swizzled scale bytes exact parity over the v1 grid | pass |

## Notes

- VLA QKV reference computes normalization and RoPE in FP32 and converts to
  BF16 only at the final output, matching the CUDA kernel.
- The older VLA QKV speedup table with max absolute error up to `0.25` remains
  invalidated as benchmark evidence.
- Small-M GEMM reports absolute error as well as BF16 ULP. Large absolute
  differences on large outputs can be a small BF16 ULP count; the built
  artifact release gate uses `max_ulp <= 5` for the BF16 matvec contract.
- BF16 GEMM epilogue wrappers in `flashrt-gemm-epilogues` are not headline
  kernels for v1. The v1 correctness claim for that package is the FP8 quant
  epilogue surface.
