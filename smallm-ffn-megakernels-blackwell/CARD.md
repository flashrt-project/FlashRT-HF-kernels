# Kernel card

## Supported operators

The callable functions and dimensional contracts are listed in the README.
Weights and FP8 inputs use contiguous E4M3FN `(N,K)` storage; biases, gates,
residuals and outputs are contiguous BF16; channel inverse-scale tensors are
also BF16, matching the original FlashRT calibration contract.
Scale floats follow the FlashRT static per-tensor calibration contract.

These are region megakernels, not arbitrary-shape linear layers. A mismatched
dimension, dtype, layout, capacity or device raises an error. No silent eager
fallback is present. CUDA 12.8+ and SM120/SM121 are required. Inference only.
