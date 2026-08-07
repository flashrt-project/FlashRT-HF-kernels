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

## SM110 (Thor) Triton note

Triton bundles its own `ptxas` (e.g. CUDA 12.8 build) which does not accept
`--gpu-name=sm_110a`, so any Triton kernel fails to compile on Thor with
`ptxas fatal: Value 'sm_110a' is not defined for option 'gpu-name'`. Point
Triton at a ptxas that knows `sm_110a` (the system CUDA 13.2 `ptxas` works):

```bash
ln -sf /usr/local/cuda-13.2/bin/ptxas \
  <venv>/lib/python3.10/site-packages/triton/backends/nvidia/bin/ptxas
# or per-invocation:
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas python ...
```

Without this, Triton-based paths (e.g. MiniMaxAI-msa-blackwell decode/indexer
tests) fail to JIT on SM110 even though the native CUDA kernels are correct.

## Dependency Policy

- Do not reuse FlashRT editable install state as a hidden dependency.
- Do not rely on `../official/FlashRT/third_party`.
- Declare CUTLASS, Torch, and Python dependencies in each package
  `build.toml`.
- Keep source copied into package-local directories so builds are reproducible.
