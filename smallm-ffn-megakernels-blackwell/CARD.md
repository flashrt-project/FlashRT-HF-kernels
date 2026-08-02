# Kernel card

## Supported operators

The callable functions and dimensional contracts are listed in the README.
Weights and FP8 inputs use contiguous E4M3FN `(N,K)` storage; biases, gates,
residuals and outputs are contiguous BF16; channel inverse-scale tensors are
also BF16, matching the original FlashRT calibration contract.
Scale floats follow the FlashRT static per-tensor calibration contract.

These are region megakernels, not arbitrary-shape linear layers. A mismatched
dimension, dtype, layout, capacity or device raises an error. No silent eager
fallback is present. SM110 uses standard FP8 MMA and explicit ordered launches
to avoid the SM120 software-grid-barrier residency assumption. SM120 retains
the original tuned path. Inference only.
