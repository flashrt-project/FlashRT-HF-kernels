# Validation

Validation requirements before publishing:

- Source correctness for smoke and full shape grids.
- Installed-artifact correctness after HF Jobs upload.
- Metrics recorded for `max_abs` where applicable, exact argmax equality, exact
  accepted-prefix equality, dtype, and unsupported-shape rejection.
- Benchmark rows should report eager PyTorch `argmax` / Python accept-prefix
  baselines separately from FlashRT kernel timings.

No public performance claim should be made from random one-off model runs.
