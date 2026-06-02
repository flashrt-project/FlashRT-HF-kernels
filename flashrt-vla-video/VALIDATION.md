# Validation: flashrt-vla-video

Validated on June 2, 2026.

Environment:

- GPU: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- PyTorch: 2.9.1+cu128
- CUDA runtime reported by PyTorch: 12.8

## Commands

Config check:

```bash
/home/heima/suliang/PI/.hf-kernel-env/bin/kernel-builder-docker check-config .
```

Result: passed.

Local source-extension compile:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python <local torch extension load smoke>
```

Result: passed for:

- `torch-ext/torch_binding.cpp`
- `csrc/q_norm_rope_bf16.cu`

Correctness smoke:

- Shapes: `(1,128)`, `(8,128)`, `(2,4,128)`, `(48,128)`
- `q_norm_rope_bf16`: max error 0 in the smoke run.
- `k_norm_rope_v_cache_bf16`: max K error 0 and V copy exact in the smoke run.
- `qkv_split_norm_rope_bf16`: package-local source extension passed launch and
  reference comparison for token lengths 4, 64, 256, 1024, and 2520.

Package-local source benchmark:

| Shape | Q fused us | Q eager us | Q speedup | K fused us | K eager us | K speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heads=1 | 2.639 | 71.105 | 26.95x | 2.564 | 73.607 | 28.71x |
| heads=8 | 2.464 | 71.944 | 29.20x | 2.555 | 74.246 | 29.06x |
| heads=48 | 2.454 | 75.614 | 30.81x | 2.667 | 77.466 | 29.05x |

Package-local QKV split + norm + RoPE benchmark:

| Shape | Tile | Fused us | PyTorch eager us | Speedup | Max error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| B=1,T=1,H=24,D=128 | 512 | 4.155 | 162.573 | 39.13x | 0.06250 |
| B=1,T=4,H=24,D=128 | 512 | 4.149 | 165.208 | 39.82x | 0.12500 |
| B=1,T=16,H=24,D=128 | 512 | 4.157 | 164.253 | 39.51x | 0.12500 |
| B=1,T=64,H=24,D=128 | 512 | 4.158 | 165.057 | 39.69x | 0.12500 |
| B=1,T=256,H=24,D=128 | 256 | 6.193 | 161.163 | 26.02x | 0.12500 |
| B=1,T=1024,H=24,D=128 | 256 | 12.131 | 235.017 | 19.37x | 0.25000 |
| B=1,T=2520,H=24,D=128 | 256 | 20.546 | 506.212 | 24.64x | 0.12500 |
| B=1,T=4096,H=24,D=128 | 256 | 36.022 | 1043.616 | 28.97x | 0.12500 |

## Remaining Gaps

- Full `kernel-builder build` has not been run for this package yet.
- Hub benchmark runner has not been run for `benchmarks/benchmark_q_norm_rope.py`.
- A downstream HF-style model-block example is still needed before making this
  the first showcase update.
- Runtime validation is currently RTX 5090 only.
