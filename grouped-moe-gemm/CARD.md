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

Validated 35B-A3B prefill shapes are `N=1024,K=2048` for fused gate/up and
`N=2048,K=512` for down projection. The caller must produce stable
expert-sorted tiles and preserve the token-to-tile row map for deterministic
weighted unpermute. Those scheduling tensors are intentionally outside this
compute-only ABI.

No hidden dequantization or eager fallback occurs in this package.

## Architecture support

- `sm_120a`/`sm_121`: native CUTLASS block-scaled MMA tiles (`M16`, `M64`,
  `64x64` block tile). This is the only dispatch target on SM120.
- `sm_110a` (Jetson AGX Thor): a portable pure-SIMT reference
  (`portable_moe_simt.cu`) computes the same grouped FP4 x FP4 -> BF16 GEMM.
  It is a compatibility path, not a performance kernel.

The dispatcher selects the SIMT reference only on non-SM120 devices where no
tensor-core backend exists. Set `FLASHRT_FORCE_SIMT=1` to route any device
through the SIMT reference (used by the correctness test to validate parity
against the native path).
