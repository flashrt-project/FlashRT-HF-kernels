# Validation

Current local validation was run on:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Runtime: local Torch 2.11 / CUDA 12.8 environment

## Correctness

Command:

```bash
python sageattention2-blackwell/tests/test_sageattention2_blackwell.py --backend source --mode full
```

Covered rows:

| Workload | Shape | Mask | V path | Result |
|---|---|---|---|---|
| Wan/video self-attn | `B=1,S=128,H=24,D=128` | none | FP16 V | PASS |
| Qwen prefill GQA | `B=1,S=128,Hq=32,Hkv=8,D=128` | causal | FP16 V | PASS |
| Wan/video self-attn | `B=1,S=256,H=24,D=128` | none | FP8 V | PASS |
| Qwen prefill GQA | `B=1,S=256,Hq=32,Hkv=8,D=128` | causal | FP8 V | PASS |
| Qwen prefill GQA | `B=1,S=512,Hq=32,Hkv=8,D=128` | causal | FP16 V | PASS |
| Wan partial tile | `B=1,S=3600/5070,H=24,D=128` | none | FP16/FP8 V | PASS |
| Qwen partial GQA | `B=1,S=3600,Hq=32,Hkv=8,D=128` | causal | FP16 V | PASS |
| Per-thread Q/K | `S=512/3600/5070` | causal/non-causal | FP16/FP8 V | PASS |

The reference is PyTorch SDPA over the same BF16 Q/K/V tensors. Sage2 is a
quantized attention path, so validation uses cosine/p99/max error gates instead
of bit-exact equality. Local full-source run passed with cosine around
`0.9993-0.999998` depending on FP16-V vs FP8-V path.

The release gate also checks:

- combined public Q/K wrapper output against the existing per-warp producer contract;
- per-thread scales and INT8 values against SageAttention grouping/rounding;
- exact native V transpose/pad/permutation and per-channel scale contracts;
- stable pointers and bitwise replay under CUDA Graph for both granularities;
- explicit rejection of invalid shapes and undersized workspace buffers.

## Published artifact gate

Cold-cache validation loaded `flashrt/sageattention2-blackwell@v1` commit
`ca36bd86` and selected `torch211-cxx11-cu128-x86_64-linux`. The installed
full matrix passed with the same metrics as the source gate. A
`torch.compile(..., fullgraph=True)` call over the static FP8-V wrapper also
passed and was bitwise equal to eager execution.

All six published variants were statically checked for both the Python
workspace surface and compiled `quantize_v_fp8_native_bf16_d128` symbol:

- Torch 2.11: CUDA 12.8 and 13.0;
- Torch 2.12: CUDA 13.0 and 13.2;
- Torch 2.13: CUDA 13.0 and 13.2.

## Benchmark

Command:

```bash
python sageattention2-blackwell/benchmarks/benchmark.py --backend source --mode full --iters 50 --warmup 10
```

Results are recorded in `benchmarks/RESULTS.md`.

The native parity gate uses the same inputs, already-quantized buffers,
caller-owned workspaces, warmup, CUDA events, and process for both paths. On
RTX 5090 the packaged core is within 1% of the FlashRT raw core with bitwise
identical core output. The complete FP8-V wrapper is within measurement noise
of the FlashRT native producer+core path on self- and cross-attention shapes.
