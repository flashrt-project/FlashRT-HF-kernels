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

Builder metadata:

- `build.toml` present.
- `flake.nix` present.
- `flake.lock` present and aligned with the other v1 packages.

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
- `qkv_split_norm_rope_bf16`: package-local source extension passed launch, but
  previous benchmark evidence is invalidated until Q and K outputs both pass an
  accuracy-first sweep.

Package-local source benchmark:

| Shape | Q fused us | Q eager us | Q speedup | K fused us | K eager us | K speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heads=1 | 2.639 | 71.105 | 26.95x | 2.564 | 73.607 | 28.71x |
| heads=8 | 2.464 | 71.944 | 29.20x | 2.555 | 74.246 | 29.06x |
| heads=48 | 2.454 | 75.614 | 30.81x | 2.667 | 77.466 | 29.05x |

Package-local QKV split + norm + RoPE benchmark:

- Invalidated as release evidence.
- Previous max absolute error reached `0.25`.
- The public HF benchmark script verified only one output tensor before this
  validation update.
- Re-run only after Q and K outputs both report `max_abs_error`,
  `max_rel_error`, and a pre-declared pass/fail threshold.

## Remaining Gaps

- Full `kernel-builder build` has not been run for this package yet.
- Hub benchmark runner has not been run for `benchmarks/benchmark_q_norm_rope.py`.
- QKV split + norm + RoPE accuracy gate is not complete.
- A downstream HF-style model-block example exists under `examples/`, but it
  still needs to be run against a built or uploaded Hub package.
- Runtime validation is currently RTX 5090 only.
