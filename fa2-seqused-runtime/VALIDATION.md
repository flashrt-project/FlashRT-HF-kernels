# Validation contract

Release requires all of the following against the installed Hub artifact:

1. Non-causal FP16/BF16 across every 8-aligned logical D through 256, with
   full production-shape coverage for D=48/72/128 and compiled-bucket
   boundaries D=64/96/128/256.
2. MHA and GQA, partial Q/K tiles, batch 1/2, aligned padded strides, and
   explicit rejection of misaligned layouts.
3. BF16 causal attention at D=128/256, including explicit bottom-right mask
   references for rectangular query/KV lengths.
4. Device `seqused_k` with different valid lengths per batch.
5. No-split for every supported logical dimension; split-KV for every
   8-aligned D=40..128 and D=232..256; verified no-split fallback and explicit
   manual-workspace rejection for unsafe partial split tiles; rounded
   accumulator dimensions and the required LSE reset.
6. CUDA Graph capture and replay with a device length update.
7. `torch.compile(fullgraph=True)` tracing through the registered fake op.
8. Explicit errors for unsupported dtype/head-dimension/path combinations.

Metrics reported per row are maximum absolute error, mean absolute error,
p99 absolute error, cosine similarity, dtype, and tolerance. Package benchmarks
must compare the installed artifact with warmed PyTorch SDPA and the original
FlashRT implementation on the same stream and static buffers.

No hardware support is claimed until an installed artifact has executed its
full relevant shape matrix on that hardware.

## x86 Hub matrix refresh (2026-08-11)

The existing Kernel Hub `v1` Torch 2.11 CUDA 12.8 x86 artifact was loaded on
RTX 5090 and passed the complete package test suite (`206/206`). The canonical
`v1` branch currently contains Torch 2.11/2.12 x86 variants, while the default
`main` alias contains no build variants. This release refresh adds Torch 2.13
CUDA 13.0/13.2 x86 targets and must promote `v1` to `main` after all serialized
FA2 jobs finish. Each newly built artifact must rerun the installed suite; the
existing `206/206` result does not stand in for unbuilt Torch 2.13 variants.
