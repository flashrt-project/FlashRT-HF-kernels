# Benchmarks

`benchmark.py` compares the package source or installed artifact against
mathematically equivalent warmed PyTorch eager and `torch.compile` references.
Original FlashRT native-pointer parity is maintained in the ignored internal
benchmark described by `VALIDATION.md`.
