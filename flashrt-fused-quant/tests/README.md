# Tests

Planned test groups:

- `SiLU(gate) * up -> NVFP4` matches a PyTorch fake-quant reference.
- Merged and split gate/up APIs produce equivalent results.
- Residual update happens in place only when the public API explicitly says so.
- RMSNorm epsilon, dtype, and scale-factor layout behavior are covered by
  boundary-shape tests.
- Rejected dtype and shape cases.
