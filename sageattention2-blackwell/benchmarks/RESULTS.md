# sageattention2-blackwell Benchmark Results

## Native V-producer parity update

RTX 5090, published `flashrt/sageattention2-blackwell@v1` artifact
`ca36bd86`, Torch 2.11/CUDA 12.8, 10 warmup and 50 measured iterations. The FP8-V path
uses the FlashRT native two-stage V producer with caller-owned BF16 transpose
workspace. The earlier direct strided V producer is retained only as a
low-level compatibility op and is not used by the public FP8-V wrapper.

| Workload | Sq/Sk | Hq/Hkv | SDPA us | Static FP8V PW us | Static FP8V PT us | PW vs SDPA | PW/PT cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 prefill | 1024/1024 | 32/8 | 96.492 | 63.654 | 69.795 | 1.52x | 0.999998/0.999393 |
| qwen3 prefill | 4096/4096 | 32/8 | 840.331 | 388.069 | 460.062 | 2.17x | 0.999998/0.999360 |
| video self-attn | 6144/6144 | 32/32 | 3106.768 | 1375.407 | 1680.252 | 2.26x | 0.999997/0.999191 |
| video self-attn | 24576/24576 | 32/32 | 45869.180 | 18748.281 | 19741.284 | 2.45x | 0.999997/0.999216 |
| video cross-attn | 6144/1024 | 32/32 | 551.809 | 282.143 | 404.425 | 1.96x | 0.999997/0.999218 |
| video cross-attn | 24576/1024 | 32/32 | 2014.350 | 1012.648 | 1287.469 | 1.99x | 0.999997/0.999256 |

Same-process direct FlashRT-native comparison used identical inputs and timing:

| Workload | Packaged full us | FlashRT native full us | Ratio | Packaged/native cosine |
|---|---:|---:|---:|---:|
| video self-attn 6144 | 1344.846 | 1360.019 | 0.989x | 0.9999883 |
| video self-attn 24576 | 18764.397 | 18619.458 | 1.008x | 0.9999874 |
| video cross-attn 6144/1024 | 278.314 | 277.645 | 1.002x | 0.9999899 |
| video cross-attn 24576/1024 | 977.043 | 986.879 | 0.990x | 0.9999892 |

The direct comparison's core-only output is bitwise identical. Small complete
path differences come from FP8 boundary rounding between separately compiled
CUDA artifacts; both paths satisfy the same FP8 numerical contract and SDPA
correctness gates.

## Per-thread and Q/K producer release gate

RTX 5090, PyTorch 2.11 + CUDA 12.8, 5 warmup and 20 measured iterations.
All rows use caller-owned buffers. `PW` is the existing per-warp Q/K contract;
`PT` is the SageAttention-compatible per-thread contract. Attempted
single-launch PW producers were rejected from the release surface because they
were slower than the two independently tuned producers.

| Workload | Sq/Sk | Hq/Hkv | SDPA us | Static FP8V PW us | Static FP8V PT us | PT vs SDPA | PW/PT cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 prefill | 1024/1024 | 32/8 | 96.701 | 74.666 | 82.221 | 1.18x | 0.999998/0.999393 |
| qwen3 prefill | 4096/4096 | 32/8 | 838.680 | 422.886 | 497.157 | 1.69x | 0.999998/0.999360 |
| video self-attn | 6144/6144 | 32/32 | 3092.154 | 1692.186 | 2021.234 | 1.53x | 0.999997/0.999191 |
| video self-attn | 24576/24576 | 32/32 | 45854.745 | 19652.870 | 20790.935 | 2.21x | 0.999997/0.999216 |
| video cross-attn | 6144/1024 | 32/32 | 549.734 | 327.395 | 449.054 | 1.22x | 0.999997/0.999218 |
| video cross-attn | 24576/1024 | 32/32 | 1991.554 | 1041.955 | 1329.354 | 1.50x | 0.999997/0.999256 |

Contract gates additionally require combined PW wrapper results to be bitwise
equal to the legacy producers, PT INT8/scales to match the official grouping
and rounding formula, partial Q tiles (`S=3600/5070`), causal GQA, F16V/FP8V,
and bitwise CUDA Graph replay for both granularities.

## PR #172 static-workspace source gate

RTX 5090, PyTorch 2.11.0+cu128, 5 warmup and 20 measured iterations. The
static F16V/FP8V paths use caller-owned workspaces; allocation is outside
timing. This source gate supplements, rather than replaces, the published
artifact table below.

| Workload | Sq/Sk | Hq/Hkv | Mask | SDPA us | Core us | Static F16V us | Static FP8V us | FP8V vs SDPA | Core cosine | p99 abs |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 prefill | 1024/1024 | 32/8 | causal | 96.790 | 67.766 | 84.123 | 73.882 | 1.31x | 0.999998 | 0.000244 |
| qwen3 prefill | 4096/4096 | 32/8 | causal | 840.448 | 504.120 | 558.125 | 423.051 | 1.99x | 0.999998 | 0.000122 |
| video self-attn | 6144/6144 | 32/32 | none | 3094.739 | 1836.022 | 2017.475 | 1691.712 | 1.83x | 0.999997 | 0.000031 |
| video self-attn | 24576/24576 | 32/32 | none | 45859.775 | 27515.961 | 28280.414 | 19638.545 | 2.34x | 0.999997 | 0.000015 |
| video cross-attn | 6144/1024 | 32/32 | none | 553.312 | 317.989 | 390.594 | 329.637 | 1.68x | 0.999997 | 0.000122 |
| video cross-attn | 24576/1024 | 32/32 | none | 2019.397 | 1155.619 | 1390.835 | 1046.165 | 1.93x | 0.999997 | 0.000122 |

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
