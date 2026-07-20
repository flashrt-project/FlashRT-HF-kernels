# Validation

Release requires correctness against an FP32 SDPA reference after dequantizing
the exact FP8 inputs. Report max, p99, mean absolute error and cosine similarity
for every supported sequence in the test matrix. Tests also cover invalid
contracts, preallocated output, `torch.compile(fullgraph=True)`, CUDA Graphs,
and a fresh-process installed-artifact import.

Performance is compared with PyTorch eager and compiled SDPA using identical
FP8 inputs and output allocation policy. Packaged performance must also match
the original FlashRT entry point within measurement noise.
