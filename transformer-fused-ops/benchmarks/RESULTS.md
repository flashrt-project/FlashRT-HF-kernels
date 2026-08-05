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

## MoE weighted sum

Source-extension benchmark on RTX 5090, PyTorch 2.9.1+cu128.

| Shape | Eager us | Compile us | Wrapper us | Raw native us |
|---|---:|---:|---:|---:|
| `tokens17 topk8 hidden2048 stride2112` | 19.616 | 23.523 | 6.155 | 6.149 |

The wrapper is 3.19x faster than eager, 3.82x faster than compile, and within
0.1% of the raw FlashRT native entry.

## NVIDIA Thor installed artifact

- Device: NVIDIA Thor, SM110, aarch64
- Runtime: PyTorch 2.13.0 + CUDA 13.0
- Variant: `torch213-cxx11-cu130-aarch64-linux`
- Native source: FlashRT `24df793f4fa2d50780aea03b644208c6e0cb4162`

The GROOT N1.7 FP16 vector-family installed gate passed 38/38. A representative
LayerNorm wrapper/raw measurement was `5.0586/5.0558 us` (`1.0006x`) in direct
mode and `2.7327/2.7282 us` (`1.0016x`) under CUDA Graph. Correctness and
static-buffer semantics are gated independently in `VALIDATION.md`.
