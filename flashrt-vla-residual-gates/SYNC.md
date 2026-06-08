# Source Sync

Synced from `official/FlashRT`:

- `csrc/kernels/elementwise.cu`: `motus_joint_residual3_out_*` family.
- `csrc/kernels/elementwise.cuh`: public pointer-level contracts.

Local adaptation:

- Public package/API names use generic VLA terminology instead of model-specific
  `motus` names.
- Only the VLA joint residual/gate subset is included in this package.
- Tensor-facing `torch.ops` bindings validate dtype, shape, contiguity, device,
  and even hidden dimensions.
- CUDA math uses explicit `__fadd_rn` and `__fmul_rn` for the PyTorch eager
  reference contract; this avoids FMA-induced one-ULP BF16 differences in
  correctness sweeps.
