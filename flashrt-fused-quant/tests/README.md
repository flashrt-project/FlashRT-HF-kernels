# Tests

Current test groups:

- `SiLU(gate) * up -> NVFP4` matches a PyTorch fake-quant reference.
- Merged and split gate/up APIs produce equivalent results.
- Rejected dtype and shape cases.

Queued test groups for later source slices:

- Residual update happens in place only when the public API explicitly says so.
- RMSNorm epsilon, dtype, and scale-factor layout behavior are covered by
  boundary-shape tests.
