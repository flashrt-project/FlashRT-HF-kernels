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

## NHD per-head sigmoid gate

RTX 5090 source gate, 10 warmup and 50 measured iterations. The reference
uses the exact BF16 contract: `2*sigmoid(gate)` rounded to BF16, followed by
BF16 activation multiplication. Outputs are bit-exact.

| Shape | Kernel us | Eager us | Speedup |
|---|---:|---:|---:|
| `S6144 H32 D128` | 55.428 | 309.930 | 5.59x |
| `S24576 H32 D128` | 276.568 | 1334.435 | 4.82x |
| `S2688 H32 D64` | 8.263 | 42.948 | 5.20x |

## NVIDIA Thor installed artifact

- Device: NVIDIA Thor, SM110, aarch64
- Runtime: PyTorch 2.13.0 + CUDA 13.0
- Variant: `torch213-cxx11-cu130-aarch64-linux`
- Native source: FlashRT `24df793f4fa2d50780aea03b644208c6e0cb4162`

The GROOT N1.7 FP16 vector-family installed gate passed 38/38. A representative
LayerNorm wrapper/raw measurement was `5.0586/5.0558 us` (`1.0006x`) in direct
mode and `2.7327/2.7282 us` (`1.0016x`) under CUDA Graph. Correctness and
static-buffer semantics are gated independently in `VALIDATION.md`.

## SM110 BF16 Producer Update

Source benchmark on NVIDIA Thor, PyTorch 2.13.0+cu130, August 8, 2026:

| Operation | Shape | Kernel us | Eager us | Speedup | Max abs | P99 abs | Cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| static FP8 quantize | `768x4304` | 18.459 | 225.205 | 12.20x | 0 | 0 | 1.0000001 |
| LayerNorm to FP8 | `512x1152` | 14.353 | 85.703 | 5.97x | 0 | 0 | 1.0000000 |
| merged GeGLU to FP8 | `768x6912` | 71.646 | 870.980 | 12.16x | 0 | 0 | 1.0000000 |

The full source gate passed `54/54`, including non-vector-aligned tails,
PI0.5/SigLIP shapes, `torch.compile(fullgraph=True)`, and CUDA Graph replay.

## GROOT Thor explicit vector entries

Clean installed artifact built from source commit
`6ab0803f010b2af49474577dddef866938c3bdcd` on NVIDIA Thor (SM110), PyTorch
2.13.0+cu130. Each explicit `*_vec` entry was measured against the established
package entry that dispatches the same FlashRT native implementation. Timings
use preallocated buffers, CUDA Graph replay, and an A-B-B-A measurement order.

| Operation | Shape | Established us | Explicit us | Explicit/established |
|---|---:|---:|---:|---:|
| RMSNorm FP16 | `41x1536` | 10.277 | 10.253 | 0.998x |
| LayerNorm FP16 | `41x1536` | 11.257 | 11.495 | 1.021x |
| LayerNorm to FP8 | `41x1536` | 12.810 | 12.796 | 0.999x |
| Static FP8 quantize | `41x1536` | 5.957 | 5.462 | 0.917x |
| Rotate-half RoPE | `41x8x128` | 6.171 | 6.152 | 0.997x |
| Residual add FP16 | `41x1536` | 3.262 | 3.050 | 0.935x |
| Repeat GQA heads | `41x4x128`, factor 2 | 6.180 | 6.153 | 0.996x |

All seven explicit entries passed bitwise alias parity. The source and clean
installed-artifact gates both passed `67/67`; the explicit RMSNorm and static
FP8 quantize calls also passed `torch.compile(fullgraph=True)`.

## RMSNorm, gated SiLU, and NVFP4 producer

RTX 5090 source-extension benchmark, PyTorch `2.11.0+cu128`, shape
`48x128`, caller-owned outputs, 20 warmups and 100 measured launches. The
staged path uses this package's RMSNorm-gated-SiLU output followed by the
production `fp4-gemm` NVFP4 quantizer; it is not a Python emulation.

| Entry | Fused us | Staged native us | Speedup | BF16 output | Packed FP4 | SFA |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `rms_norm_gated_silu_quant_fp4_bf16` | 4.10 | 6.07 | 1.48x | exact | exact | exact |

The release gate also requires bit-identical CUDA Graph replay and
`torch.compile(fullgraph=True)` execution from the installed Hub artifact.
