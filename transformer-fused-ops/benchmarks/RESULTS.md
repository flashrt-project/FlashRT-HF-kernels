# Results

Source-extension benchmark on NVIDIA GeForce RTX 5090. Correctness remains
gated independently in `VALIDATION.md`.

## ReLU2 to FP8 producer

The original Cosmos Edge launcher and the generic Tensor wrapper are bitwise
identical. CUDA Graph rows capture 32 launches and report time per launch.

| Shape | Native us | Wrapper us | Wrapper/native | Graph native us | Graph wrapper us | Graph wrapper/native | Eager us | Compile us | FP8 exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `277x2048` | 2.058 | 2.328 | 1.131 | 1.407 | 1.408 | 1.000 | 19.087 | 21.437 | yes |

The direct 0.27 us delta is host dispatch overhead. The graph hot path is at
native parity.
