# Validation contract

Release requires all of the following against the installed Hub artifact:

1. Non-causal FP16/BF16 across D=64/96/128/256.
2. MHA and GQA, partial Q/K tiles, batch 1/2, aligned padded strides, and
   explicit rejection of misaligned layouts.
3. BF16 causal attention at D=128/256, including explicit bottom-right mask
   references for rectangular query/KV lengths.
4. Device `seqused_k` with different valid lengths per batch.
5. No-split and split-KV paths, including the required LSE reset.
6. CUDA Graph capture and replay with a device length update.
7. `torch.compile(fullgraph=True)` tracing through the registered fake op.
8. Explicit errors for unsupported dtype/head-dimension/path combinations.

Metrics reported per row are maximum absolute error, mean absolute error,
p99 absolute error, cosine similarity, dtype, and tolerance. Package benchmarks
must compare the installed artifact with warmed PyTorch SDPA and the original
FlashRT implementation on the same stream and static buffers.

No hardware support is claimed until an installed artifact has executed its
full relevant shape matrix on that hardware.
