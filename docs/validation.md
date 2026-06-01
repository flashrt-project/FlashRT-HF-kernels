# Validation Checklist

Run this checklist before promoting any package from draft to buildable.

## Layout

- Package has `README.md`, `CARD.md`, `build.toml`, `flake.nix`, `tests/`,
  `benchmarks/`, `torch-ext/`, and `csrc/`.
- Public package name under `torch-ext/` matches `[general].name` with dashes
  replaced by underscores.
- `build.toml` lists every required source file and header.
- No build outputs are committed.

## Native Binding

- Native functions are registered with `TORCH_LIBRARY_EXPAND`.
- `REGISTER_EXTENSION(TORCH_EXTENSION_NAME)` is present.
- Python wrappers import `ops` from `._ops`.
- Python-defined Torch custom ops use `add_op_namespace_prefix`.
- Public APIs accept `torch.Tensor` objects rather than raw pointers.
- Device, dtype, shape, stride, contiguity, and output aliasing rules are
  checked in C++ or Python before launch.

## Correctness

- Tests compare against PyTorch reference output when feasible.
- Tests in package `tests/` do not require `../official/FlashRT` or private
  model fixtures.
- FlashRT parity tests that require upstream runtime state live under
  `internal-tests/`.
- Tests cover supported dtypes.
- Tests cover non-default CUDA device when a machine has multiple GPUs.
- Tests cover rejected shapes and dtypes.
- Numerical tolerances are documented.
- Tests are marked so a small CI subset can run in under 60 seconds.

## Performance

- Benchmarks include generic shapes that are understandable outside FlashRT.
- Benchmarks include one FlashRT-real shape family.
- Benchmarks report latency, bandwidth or TFLOPS where relevant, and speedup
  versus PyTorch eager or a known baseline.
- Benchmark scripts pin warmup/iteration counts and synchronize CUDA timing.

## Build and ABI

- `kernel-builder build <package>` succeeds.
- `kernel-builder check-abi <package>` succeeds.
- CUDA architecture constraints are represented in `build.toml`.
- CUDA min/max versions are set only when required.
- CUTLASS or other dependencies are declared through `depends` or
  `python-depends`; package builds do not depend on FlashRT `third_party`.

## Hub Readiness

- `CARD.md` describes supported hardware, dtype, shapes, and limitations.
- `README.md` shows a minimal `get_kernel` example.
- Version is bumped only for API-breaking changes or incompatible build
  variant changes.
- License is declared and compatible with copied upstream source.
