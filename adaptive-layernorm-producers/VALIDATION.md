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
| `ada_layer_norm_quant_fp8_ptok_bf16` | per-token FP8 reference contract |
| `ada_layer_norm_quant_fp8_ptok_table_bf16` | fused table-add/chunk-select FP8 reference contract |
| `ada_layer_norm_quant_fp8_modfp8_bf16` | FP8 reference contract |
| `awq_ada_layer_norm_quant_fp8_bf16` | FP8 reference contract |
| `layer_norm_no_affine_quant_fp8_static_bf16` | FP8 reference contract |
| `ada_layer_norm_quant_nvfp4_swizzled_bf16` | exact packed output and exact swizzled scale-factor output for representative rows |
| `ada_layer_norm_quant_nvfp4_swizzled_modfp8_bf16` | exact packed output and exact swizzled scale-factor output for representative rows |
| `adaln_modulation6_bf16` | exact six-output BF16 parity, fullgraph compile, raw native and CUDA Graph benchmark |

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

## NVIDIA aarch64 release gate

The `torch211-cxx11-cu130-aarch64-linux` artifact contains native SM87 and
SM110a code objects. Its SM110a path is tested on NVIDIA Thor with the same
full matrix. The two per-token producer entries additionally require:

- direct `(rows, dim)` and table `(rows, chunks, dim)` modulation coverage;
- bit-exact installed-wrapper versus raw registered-op output;
- A-B-B-A timing on M51/D1536, M105/D1152, and M2520/D3072;
- CUDA Graph replay; and
- successful loading through the current Kernel Hub client and the legacy
  `kernels<0.13` model mirror.

The exact artifact downloaded from Kernel Hub `v1` passed all full-test rows
on SM110a and all six per-token raw/wrapper rows. The measured wrapper/raw
range was `0.997-1.000`. `cuobjdump --list-elf` confirms native `sm_87` and
`sm_110a` cubins in the published shared object. An Orin runtime gate is still
required before claiming SM87 execution or performance validation.
