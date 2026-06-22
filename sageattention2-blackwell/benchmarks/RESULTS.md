# sageattention2-blackwell Benchmark Results

Published-artifact benchmark:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Runtime: local Torch 2.11 / CUDA 12.8 environment
- Artifact: `flashrt/sageattention2-blackwell@v1`, `torch211-cxx11-cu128-x86_64-linux`
- Source commit in artifact name: `1556b76`
- Command: `python sageattention2-blackwell/benchmarks/benchmark.py --backend installed --artifact <artifact> --mode full --iters 50 --warmup 10`
- Baseline: PyTorch SDPA with the same BF16 Q/K/V tensors and mask mode.
- `Sage core`: already-quantized Q/K/V input path.
- `BF16 wrapper`: public convenience path including Q/K/V quantization.
- Coverage includes partial Q-tile shapes such as Wan `S=5070` (`S % 128 = 78`).

| Workload | S | Hq/Hkv | Mask | SDPA us | Sage core us | Core speedup | BF16 wrapper us | Wrapper speedup | Cos | p99 abs |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3_prefill | 1024 | 32/8 | causal | 95.456 | 68.024 | 1.40x | 84.591 | 1.13x | 0.999998 | 0.000244 |
| qwen3_prefill | 2048 | 32/8 | causal | 258.387 | 179.355 | 1.44x | 205.866 | 1.26x | 0.999998 | 0.000244 |
| qwen3_prefill | 4096 | 32/8 | causal | 839.730 | 521.421 | 1.61x | 576.975 | 1.46x | 0.999998 | 0.000122 |
| qwen3_prefill | 8192 | 32/8 | causal | 2927.033 | 1796.252 | 1.63x | 1895.960 | 1.54x | 0.999997 | 0.000122 |
| wan_self_attn | 1024 | 24/24 | none | 110.818 | 69.722 | 1.59x | 98.464 | 1.13x | 0.999997 | 0.000122 |
| wan_self_attn | 2520 | 24/24 | none | 394.460 | 243.882 | 1.62x | 302.475 | 1.30x | 0.999997 | 0.000061 |
| wan_self_attn | 4096 | 24/24 | none | 1044.623 | 640.698 | 1.63x | 739.503 | 1.41x | 0.999997 | 0.000061 |
| wan_self_attn | 5070 | 24/24 | none | 1567.687 | 951.150 | 1.65x | 1075.493 | 1.46x | 0.999997 | 0.000061 |
