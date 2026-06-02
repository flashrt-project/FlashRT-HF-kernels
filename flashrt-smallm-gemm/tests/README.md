# Tests

Planned test groups:

- W4A4 matvec correctness for `M=1` decode shapes against a dequantized PyTorch
  reference.
- W4A4 warpsplit correctness for `M in {1, 2, 4, 8, 16}`.
- Tiny FP8 fixed-family correctness for each exposed shape.
- Runtime guards reject unsupported M/K/N/layout combinations with clear errors.
