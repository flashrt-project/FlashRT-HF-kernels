# Contributing

This repository follows the Hugging Face Kernel Hub project layout, with an
extra separation between public package material and local FlashRT maintenance
material.

## Public Package Material

Public package material belongs in a package directory such as
`flashrt-gemm-epilogues/`.

This includes:

- `README.md`
- `CARD.md`
- `SYNC.md`
- `build.toml` or `build.toml.draft`
- `csrc/`
- `torch-ext/`
- `tests/`
- `benchmarks/`

Package `tests/` should be portable. They should not require the adjacent
FlashRT checkout, private model fixtures, or private hardware setup.

## Internal Material

Use `internal-docs/` for local planning, open design questions, and source
selection notes.

Use `internal-tests/` for tests that compare against the FlashRT upstream
runtime or use model-derived fixtures. These tests are useful for maintainers,
but they are not Hub package CI.

## Build Files

Use `build.toml.draft` while a package is being scoped. Rename it to
`build.toml` only after all listed sources exist and the package is expected to
build with `kernel-builder`.

## Naming

Prefer generic names that describe math, dtype, layout, and shape behavior.
Avoid model names in public package names and APIs unless the package is
explicitly a model compatibility layer.

## Validation

Before publishing or proposing a package for `kernels-community`, complete the
checklist in `docs/validation.md`.
