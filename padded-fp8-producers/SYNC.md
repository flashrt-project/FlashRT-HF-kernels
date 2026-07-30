# Source synchronization

## Provenance

The initial implementation is derived from the production FlashRT kernels:

- `csrc/kernels/lingbot_norm_fp8.cu`
- `csrc/kernels/lingbot_silu_mul_fp8.cu`
- `csrc/kernels/lingbot_common.cuh`

The exact upstream commit must be recorded at release time.

## Packaging changes

- Removed model-specific names and raw `uintptr_t` public APIs.
- Added Tensor validation, current-stream dispatch, device guards, fake
  registration, and preallocated-output Python APIs.
- Generalized contracts to arbitrary positive batch, row, padded-row, and
  feature dimensions.
- The residual path accumulates RMS statistics from the BF16 residual value
  written to `residual_out`. The original source accumulated the pre-rounding
  FP32 sum. This intentional correction aligns the public BF16 eager contract
  and removes avoidable FP8-bin drift.
- No dynamic allocation occurs in CUDA launchers.

## Unsupported

- Non-contiguous tensors.
- `padded_rows < rows`.
- Dynamic or per-token scales.
- FP8 formats other than E4M3FN.
