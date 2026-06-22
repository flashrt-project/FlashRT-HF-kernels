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

The reference is PyTorch SDPA over the same BF16 Q/K/V tensors. Sage2 is a
quantized attention path, so validation uses cosine/p99/max error gates instead
of bit-exact equality. Local full-source run passed with cosine around
`0.9993-0.999998` depending on FP16-V vs FP8-V path.

## Benchmark

Command:

```bash
python sageattention2-blackwell/benchmarks/benchmark.py --backend source --mode full --iters 50 --warmup 10
```

Results are recorded in `benchmarks/RESULTS.md`.
