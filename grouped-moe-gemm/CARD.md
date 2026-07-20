# Kernel card

## Callable API

`grouped_nvfp4_gemm_bf16(input, weight, input_scale, weight_scale, alpha, tile_expert, *, tile_rows, input_scale_stride=0, weight_stride=None, weight_scale_stride=None, out=None)`

- `input`: packed E2M1 `uint8 [num_tiles*tile_rows,K/2]`.
- `weight`: packed E2M1 `uint8 [experts,N,K/2]`.
- scale tensors: flat or expert-stacked CUTLASS Sm1xx swizzled UE4M3 bytes.
- `alpha`: FP32 global scale per expert.
- `tile_expert`: INT32 expert index per tile; `-1` denotes an unused tile.
- output: BF16 `[num_tiles*tile_rows,N]`.
- `K` must be divisible by 64. `tile_rows=16` requires `N%8==0`.
  `tile_rows=64` dispatches the 64x64 block tile when `N%64==0`, otherwise
  the M64/N16 tile and requires `N%16==0`.
- CUDA 12.8+, SM120/SM121. Inference only.

No hidden dequantization or eager fallback occurs in this package.
