# sageattention2-blackwell Benchmark Results

Published-artifact benchmark:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 580.82.07
- Runtime: local Torch 2.11 / CUDA 12.8 environment
- Artifact: `flashrt/sageattention2-blackwell@v1`, `torch211-cxx11-cu128-x86_64-linux`
- Command: `python sageattention2-blackwell/benchmarks/benchmark.py --backend installed --artifact <artifact> --mode full --iters 50 --warmup 10`
- Baseline: PyTorch SDPA with the same BF16 Q/K/V tensors and mask mode.
- `Sage core`: already-quantized Q/K/V input path.
- `BF16 wrapper`: public convenience path including Q/K/V quantization.

| Workload | S | Hq/Hkv | Mask | SDPA us | Sage core us | Core speedup | BF16 wrapper us | Wrapper speedup | Cos | p99 abs |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3_prefill | 1024 | 32/8 | causal | 95.398 | 67.681 | 1.41x | 84.112 | 1.13x | 0.999998 | 0.000244 |
| qwen3_prefill | 2048 | 32/8 | causal | 258.427 | 179.139 | 1.44x | 205.681 | 1.26x | 0.999998 | 0.000244 |
| qwen3_prefill | 4096 | 32/8 | causal | 839.829 | 522.608 | 1.61x | 575.830 | 1.46x | 0.999998 | 0.000122 |
| qwen3_prefill | 8192 | 32/8 | causal | 2915.043 | 1794.989 | 1.62x | 1893.889 | 1.54x | 0.999997 | 0.000122 |
| wan_self_attn | 1024 | 24/24 | none | 110.822 | 67.786 | 1.63x | 96.684 | 1.15x | 0.999997 | 0.000122 |
| wan_self_attn | 2520 | 24/24 | none | 394.952 | 240.941 | 1.64x | 298.605 | 1.32x | 0.999898 | 0.000366 |
| wan_self_attn | 4096 | 24/24 | none | 1045.161 | 630.415 | 1.66x | 727.708 | 1.44x | 0.999997 | 0.000061 |
