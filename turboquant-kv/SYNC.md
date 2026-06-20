# Source Sync

This package implements standalone tensor APIs based on FlashRT TurboQuant
sources:

- `official/FlashRT/csrc/quantize/tq_dequant_kv.cu`
- `official/FlashRT/csrc/bindings.cpp`

Only the strictly validated unpack/combine subset is exposed in this Hub
package. Write-side packing and CUTLASS/cuBLAS GEMM orchestration should be
added only after separate package-level and model-level validation.
