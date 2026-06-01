# Benchmarks

`benchmark.py` compares the first fused epilogue slice against the equivalent
PyTorch eager expression.

`benchmark_channel_scale.py` compares per-channel scaling plus FP8 quantization
against the equivalent PyTorch eager expression.

Planned benchmark expansions:

- Generic Transformer FFN projection shapes.
- Small batch and decode shapes.
- VLA/DiT projection shapes.
- Full GEMM plus fused epilogue once GEMM wrappers are synced.
- FlashRT internal baseline for local regression only.
