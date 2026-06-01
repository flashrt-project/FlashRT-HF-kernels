# Environment

This repository should be developed in an environment independent from the
FlashRT serving runtime.

## Install Kernel Builder

Follow the official installation instructions:

```bash
curl -fsSL https://raw.githubusercontent.com/huggingface/kernels/main/install.sh | bash
```

This installs Nix and `kernel-builder`.

## Local Development

For a buildable package:

```bash
kernel-builder build flashrt-gemm-epilogues
kernel-builder check-abi flashrt-gemm-epilogues
nix run ./flashrt-gemm-epilogues#ci-test
```

For local Python experiments, use `kernels.get_local_kernel` after the package
has a build result.

## Internal Tests

Internal tests may depend on the adjacent FlashRT checkout:

```bash
PYTHONPATH=../official/FlashRT pytest internal-tests
```

These tests are for source sync confidence and FlashRT parity. They are not
Hub-compatible CI tests and should not be copied into package `tests/`.

## Dependency Policy

- Do not reuse FlashRT editable install state as a hidden dependency.
- Do not rely on `../official/FlashRT/third_party`.
- Declare CUTLASS, Torch, and Python dependencies in each package
  `build.toml`.
- Keep source copied into package-local directories so builds are reproducible.
