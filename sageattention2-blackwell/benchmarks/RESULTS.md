# sageattention2-blackwell Benchmark Results

Local source-build benchmark:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Runtime: local Torch 2.11 / CUDA 12.8 environment
- Command: `python sageattention2-blackwell/benchmarks/benchmark.py --backend source --mode full --iters 50 --warmup 10`
- Baseline: PyTorch SDPA with the same BF16 Q/K/V tensors and mask mode.
- `Sage core`: already-quantized Q/K/V input path.
- `BF16 wrapper`: public convenience path including Q/K/V quantization.

| Workload | S | Hq/Hkv | Mask | SDPA us | Sage core us | Core speedup | BF16 wrapper us | Wrapper speedup | Cos | p99 abs |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3_prefill | 1024 | 32/8 | causal | 95.759 | 67.450 | 1.42x | 84.080 | 1.14x | 0.999998 | 0.000244 |
| qwen3_prefill | 2048 | 32/8 | causal | 259.213 | 163.240 | 1.59x | 192.010 | 1.35x | 0.999998 | 0.000244 |
| qwen3_prefill | 4096 | 32/8 | causal | 839.641 | 508.840 | 1.65x | 560.584 | 1.50x | 0.999998 | 0.000122 |
| qwen3_prefill | 8192 | 32/8 | causal | 2924.040 | 1737.631 | 1.68x | 1854.527 | 1.58x | 0.999997 | 0.000122 |
| wan_self_attn | 1024 | 24/24 | none | 110.804 | 67.663 | 1.64x | 96.434 | 1.15x | 0.999997 | 0.000122 |
| wan_self_attn | 2520 | 24/24 | none | 394.764 | 238.964 | 1.65x | 297.452 | 1.33x | 0.999898 | 0.000366 |
| wan_self_attn | 4096 | 24/24 | none | 1045.051 | 620.483 | 1.68x | 717.987 | 1.46x | 0.999997 | 0.000061 |
