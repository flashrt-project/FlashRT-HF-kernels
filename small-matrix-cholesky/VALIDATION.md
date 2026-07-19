# Validation status

## Completed before packaging

- The source algorithm passed all 17 public GPU MODE Cholesky property cases
  on NVIDIA A800.
- GPU MODE submission `887048` passed public and secret B200 correctness.
- The complete competition submission scored a 0.178772 ms public geometric
  mean and ranked first in the saved 2026-07-19 snapshot. That score includes
  competition-only paths and must not be attributed to this package alone.

## Package gates

- [x] Source-extension smoke correctness on A800: 11/11
- [x] Source-extension full correctness on A800: 14/14
- [x] Rejected dtype, shape, stride, alias, and device cases
- [x] Non-default CUDA device test on the second A800
- [x] Package-specific A800 benchmark: 1.521x geometric-mean speedup
- [x] Repository `prebuild_check.py` layout check
- [ ] `kernel-builder-docker check-config .`
- [ ] `kernel-builder build small-matrix-cholesky`
- [ ] `kernel-builder check-abi small-matrix-cholesky`
- [ ] Installed-artifact correctness
- [ ] Maintainer decision on a FlashRT-real benchmark family

The package uses `build.toml.draft` until the remaining promotion gates are
resolved.
