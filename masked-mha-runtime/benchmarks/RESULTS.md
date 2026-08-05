# Results

## NVIDIA Thor installed artifact

- Device: NVIDIA Thor, SM110, aarch64
- Runtime: PyTorch 2.13.0 + CUDA 13.0
- Variant: `torch213-cxx11-cu130-aarch64-linux`
- Native source: FlashRT `24df793f4fa2d50780aea03b644208c6e0cb4162`

The installed-artifact full gate passed 6/6. Representative BF16 rows:

| Shape | p99 absolute error | Cosine | CUDA Graph replay |
|---|---:|---:|---|
| `sk=1024` | 0.000452 | 0.99999362 | bitwise |
| `sk=1025` | 0.000452 | 0.99999356 | bitwise |
| `sk=2048` | 0.000174 | 0.99999666 | bitwise |

The Hub Tensor wrapper and the raw registered native op were measured in the
same process with CUDA events. Direct wrapper/raw latency was
`24.7932/24.7544 us` (`1.0016x`). With the caller-owned static buffers used by
the production CUDA Graph path, wrapper/raw latency was
`19.0034/19.0034 us` (`1.0000x`).

These numbers establish package parity with the FlashRT native symbol. They
are not presented as a universal speedup over PyTorch SDPA; backend selection
must still be profiled for the target sequence and padding profile.
