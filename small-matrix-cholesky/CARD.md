# small-matrix-cholesky

## Summary

Batched dense FP32 Cholesky factorization specialized for matrix orders 32,
64, and 128. The implementation uses one or more matrices per CUDA block and
keeps the active factorization tile in shared memory.

## API

```python
cholesky_small_fp32(input: torch.Tensor, *, out=None) -> torch.Tensor
```

## Supported inputs

- Backend: CUDA
- Dtype: FP32
- Layout: contiguous row-major dense tensor
- Shape: `(..., n, n)`, `n in {32, 64, 128}`
- Matrix property: symmetric positive definite

## Output

A tensor with the same shape, dtype, and device as the input. Each output
matrix is lower triangular with positive diagonal and reconstructs the input
as `L @ L.T` within FP32 rounding error.

## Hardware

The implementation uses standard CUDA FP32 instructions. The `n=128` path
requires at least 66,048 bytes of opt-in dynamic shared memory per block. The
draft build matrix targets CUDA compute capabilities 8.0 and newer.

## Limitations

- No fallback for unsupported sizes or insufficient shared memory.
- No status tensor is returned for non-SPD inputs; callers must uphold the SPD
  precondition.
- The package does not cache inputs or outputs and does not use reduced-
  precision Tensor Core updates.
- Package promotion remains pending a Kernel Hub builder build/ABI check and a
  maintainer decision on the absence of a current FlashRT model call site.

## License

Apache-2.0.
