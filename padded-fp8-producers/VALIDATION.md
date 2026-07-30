# Validation

Release gates:

1. Source and installed-artifact tests over logical rows
   `1, 40, 49, 51, 64, 105, 277`, padded rows
   `16, 64, 128, 320`, and feature dimensions
   `1280, 1536, 2048, 3424, 4096`.
2. Correctness reports FP8 code mismatch plus dequantized max, p99, mean, and
   cosine metrics. Tail rows must be bitwise zero.
3. Wrong dtype, non-contiguous input, malformed modulation, and
   `padded_rows < rows` must raise.
4. `torch.compile(fullgraph=True)` must preserve the custom op and output.
5. Benchmark each accepted shape against:
   - raw native CUDA launcher,
   - Hub Tensor wrapper,
   - equivalent PyTorch eager,
   - equivalent warmed `torch.compile`.
6. Wrapper/native latency must remain within measurement noise. A shape is not
   promoted based only on an eager speedup.

Current source sweep: 29/29 checks passed on RTX 5090. Installed artifact and
Hub gates remain pending until the unified release build.
