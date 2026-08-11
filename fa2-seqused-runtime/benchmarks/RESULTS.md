# Benchmark results

## Logical-head-dimension release candidate

- Split-KV qualification: logical D=40..128 and D=232..256 in steps of 8.
  Other 8-aligned D through 256 remain supported by the no-split path.
- GPU: NVIDIA GeForce RTX 5090, SM120
- Artifact: local `kernel-builder` `torch211-cxx11-cu128-x86_64-linux`
- PyTorch: `2.11.0+cu128`
- Dtype: BF16
- Timing: 50 warmups, median of 200 CUDA-event samples
- Baseline: warmed PyTorch SDPA; GQA head expansion is outside timing

| Workload | Mode | B | Sq | Sk | Hq/Hkv | D | Package us | SDPA us | Speedup | Max abs | P99 abs | Mean abs | Cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GROOT DiT self | noncausal | 1 | 51 | 51 | 32/32 | 48 | 11.008 | 14.176 | 1.288x | 0 | 0 | 0 | 0.9999998808 |
| GROOT DiT cross | noncausal | 1 | 51 | 1024 | 32/32 | 48 | 16.192 | 21.264 | 1.313x | 0.000244141 | 0 | 0.000000022 | 0.9999999404 |
| GROOT N1.7 ViT | noncausal | 1 | 256 | 256 | 16/16 | 64 | 14.976 | 18.176 | 1.214x | 0.000976562 | 0 | 0.000000049 | 1.0000000000 |
| GROOT SigLIP | noncausal | 2 | 256 | 256 | 16/16 | 72 | 20.240 | 25.312 | 1.251x | 0.001953125 | 0 | 0.000000161 | 1.0000000000 |
| VL vision | noncausal | 1 | 256 | 256 | 16/16 | 80 | 14.256 | 18.720 | 1.313x | 0.000976562 | 0 | 0.000000066 | 1.0000001192 |
| GQA decode | noncausal | 1 | 1 | 512 | 8/2 | 128 | 14.208 | 18.240 | 1.284x | 0 | 0 | 0 | 1.0000001192 |
| GQA short | noncausal | 1 | 16 | 1024 | 16/4 | 128 | 14.304 | 20.384 | 1.425x | 0.000488281 | 0 | 0.000000022 | 1.0000000000 |
| VLA GQA | noncausal | 1 | 49 | 2520 | 24/4 | 128 | 26.464 | 33.440 | 1.264x | 0.000244141 | 0 | 0.000000032 | 1.0000000000 |
| GQA long KV | noncausal | 1 | 64 | 4096 | 32/8 | 128 | 46.976 | 53.584 | 1.141x | 0.000244141 | 0 | 0.000000017 | 1.0000000000 |
| Qwen causal | causal | 1 | 1024 | 1024 | 32/8 | 128 | 86.144 | 91.872 | 1.066x | 0.001953125 | 0 | 0.000000024 | 1.0000000000 |
| Qwen3.6 causal | causal | 1 | 512 | 512 | 24/4 | 256 | 44.928 | 51.872 | 1.155x | 0.000976562 | 0 | 0.000000009 | 1.0000000000 |

The rebuilt artifact passed `154/154` tests on the same GPU. Coverage includes
every 8-aligned logical D through 256 in FP16 and BF16, production
D=48/64/72/80/128/256 rows, MHA/GQA, partial tiles, causal BF16,
split-KV including D<=64, device
`seqused_k`, CUDA Graph replay, padded aligned strides, rejection tests, and
`torch.compile(fullgraph=True)`.

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

The original 40-row detailed accuracy sweep over FP16/BF16,
D=64/96/128/256, MHA/GQA, partial tiles and causal BF16 produced:

- worst maximum absolute error vs PyTorch SDPA: `0.001953125`
- worst cosine similarity: `0.9999961853`
- package output vs original FlashRT output on benchmark rows: exact

These older rows retain the direct original-FlashRT comparison. A fresh-process
`get_kernel("flashrt/fa2-seqused-runtime", version=1)` load of the canonical
`torch211-cxx11-cu128-x86_64-linux` artifact passed all `206/206` package
tests, including the complete logical-head-dimension matrix, split-KV D<=64,
causal/GQA paths, device `seqused_k`, CUDA Graph replay, rejection contracts,
and `torch.compile(fullgraph=True)`.
