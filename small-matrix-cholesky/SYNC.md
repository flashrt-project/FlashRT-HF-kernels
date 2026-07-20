# Source provenance

## Upstream baseline

- FlashRT commit: `a4e01b0ef903643ca0f12cc5b4b914dcd3bfc4d5`
- FlashRT source files copied: none

FlashRT did not contain a Cholesky implementation at this baseline. This is a
new generic kernel developed in the isolated `FlashRT-gpumode/cholesky`
workspace for GPU MODE submission `887048` and then adapted to the Hugging
Face Kernel Hub package contract.

## Prototype source

- Prototype file: `FlashRT-gpumode/cholesky/submission.py`
- Prototype SHA-256:
  `7ee55a845f25a6de3fffdd6cc7e84d4483da91e00684a93f0652620dcda64a82`
- Relevant prototype paths: FP32 `n=32/64/128` shared-memory kernels only

## Packaging edits

- Removed all GPU MODE task imports, fixed benchmark dispatch, result
  memoization, output pools, Triton fallbacks, cuSOLVER paths, and approximate
  BF16/TF32 large-matrix paths.
- Split the CUDA implementation from the Torch Tensor binding.
- Added current-stream launch, CUDA-device guarding, dtype/shape/stride/device
  validation, output alias rejection, Python allocation helper, and fake-op
  registration.
- Added portable property-based correctness tests and a preallocated-output
  benchmark against `torch.linalg.cholesky_ex`.

## Compile and runtime assumptions

- C++17-capable PyTorch CUDA extension toolchain.
- CUDA 12.8 in the draft Kernel Hub build matrix.
- Contiguous row-major FP32 SPD matrices.
- Matrix order 32, 64, or 128.
- The `n=128` path requires 66,048 bytes of dynamic shared memory.
