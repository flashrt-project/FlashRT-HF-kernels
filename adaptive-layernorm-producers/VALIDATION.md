# Validation

Current local validation was run on:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Source runtime: local Torch 2.11 / CUDA 12.8 environment
- Package: `adaptive-layernorm-producers`

## Correctness Matrix

Command:

```bash
python adaptive-layernorm-producers/tests/test_adaptive_layernorm_producers.py --backend source --mode full
```

Covered shapes:

| Shape | Rows | Dim | Purpose |
|---|---:|---:|---|
| decode_action | 16 | 2048 | VLA/action producer |
| wan_video_short | 64 | 3072 | short video block |
| wan_video_ctx | 256 | 3072 | context/video block |
| wan_video_2k | 2520 | 3072 | Wan-style video token count |
| wan_video_4k | 4096 | 3072 | long video/world-model token count |

Covered operators:

| Operator | Check |
|---|---|
| `ada_layer_norm_quant_fp8_bf16` | FP8 reference contract |
| `ada_layer_norm_quant_fp8_modfp8_bf16` | FP8 reference contract |
| `awq_ada_layer_norm_quant_fp8_bf16` | FP8 reference contract |
| `layer_norm_no_affine_quant_fp8_static_bf16` | FP8 reference contract |
| `ada_layer_norm_quant_nvfp4_swizzled_bf16` | exact packed output and exact swizzled scale-factor output for representative rows |
| `ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16` | exact packed output and exact swizzled scale-factor output for representative rows |

FP8 long-shape validation allows only adjacent FP8-code boundary differences
caused by reference reduction/order at quantization thresholds. The gate still
requires:

- `p99_abs == 0`
- cosine similarity approximately `1.0`
- tiny nonzero count relative to output size

NVFP4 validation uses a CPU bit-level reference for E2M1 packing and the
FlashRT/CUTLASS 128x4 swizzled UE4M3 scale layout.

## Benchmark

Command:

```bash
python adaptive-layernorm-producers/benchmarks/benchmark.py --backend source --iters 100
```

Results are recorded in `benchmarks/RESULTS.md`.
