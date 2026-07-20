# Benchmark results

## Pre-release source qualification

- GPU: NVIDIA GeForce RTX 5090, SM120
- PyTorch: `2.9.0a0+145a3a7bda.nv25.10`
- CUDA: 13.0
- FlashRT source: `b3eab55`
- Dtype: BF16
- Timing: warmed CUDA events, median of 20 groups x 20 calls
- Baseline: PyTorch SDPA with GQA heads expanded before timing

| B | Sq | Sk | Hq/Hkv | D | Splits | Package us | Original FlashRT us | SDPA us | vs original | vs SDPA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 512 | 8/2 | 128 | 4 | 10.453 | 11.340 | 11.684 | 0.922x | 1.118x |
| 1 | 16 | 1024 | 16/4 | 128 | 8 | 10.477 | 11.298 | 11.712 | 0.927x | 1.118x |
| 1 | 49 | 2520 | 24/4 | 128 | 10 | 22.761 | 23.014 | 25.011 | 0.989x | 1.099x |
| 1 | 64 | 4096 | 32/8 | 128 | 7 | 43.226 | 43.461 | 43.662 | 0.995x | 1.010x |
| 1 | 1024 | 1024 | 32/8 | 128 | 1 | 108.701 | 108.928 | 111.005 | 0.998x | 1.021x |

`vs original` is `package/original`; values at or below 1.0 show no packaged
kernel regression. `vs SDPA` is `SDPA/package`. K/V head expansion is excluded
from SDPA timing, so the comparison does not credit FlashRT for avoiding that
materialization.

The 40-row detailed accuracy sweep over FP16/BF16, D=64/96/128/256, MHA/GQA,
partial tiles and causal BF16 produced:

- worst maximum absolute error vs PyTorch SDPA: `0.001953125`
- worst cosine similarity: `0.9999961853`
- package output vs original FlashRT output on benchmark rows: exact

These are source-artifact qualification numbers, not the final Hub artifact
claim. Installed `kernel-builder` results replace this section after upload and
fresh-process `get_kernel(..., version=1)` validation.
