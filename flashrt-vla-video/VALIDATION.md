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

Source accuracy sweep:

```bash
python scripts/accuracy_sweep.py --backend source --mode full --package flashrt-vla-video
```

Result: passed 110 checks.

Covered:

- `q_norm_rope_bf16`: rows `1,4,8,16,24,32,48,64,128,256`.
- `k_norm_rope_v_cache_bf16`: same rows; K output checked against BF16
  reference and V copy checked by byte parity.
- `qkv_split_norm_rope_bf16`: Q and K outputs both checked for tokens
  `1,4,16,64,256,1024,2520,4096` and heads `8,16,24,32,48`.

Accuracy contract:

- BF16 Q/K outputs: `max_abs <= 0.03125`, `max_rel <= 0.05` with
  `rel_floor=1`.
- V output: byte parity.

Worst recorded QKV source-sweep max absolute error: `0.015625`.

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
- The source accuracy sweep above replaces that correctness evidence, but the
  speedup table remains invalidated until rerun against the corrected reference
  and built package artifact.

## Remaining Gaps

- `kernel-builder build --variant torch211-cxx11-cu128-x86_64-linux` passed
  for this package, and the copied artifact passed package tests, examples,
  installed accuracy sweep, and the local release-candidate benchmark runner.
- Full `kernel-builder build-and-copy` matrix has not been run for this
  package yet.
- Official Hub `kernels benchmark` has not been run after upload.
- A downstream HF-style model-block example exists under `examples/`, but it
  still needs a real model-level throughput run.
- Runtime validation is currently RTX 5090 only.
