# Validation

Release gates cover the Higgs TTS production shape `(8,1026,hidden)` plus
non-power-of-two vocabulary and hidden boundaries. Required checks include:

- exact argmax codes, tie behavior, delay/BOC behavior and embedding sum;
- deterministic sampling against an independent SplitMix64/equal-logit oracle;
- max/p99/mean error, cosine and output dtype;
- `torch.compile(fullgraph=True)` and CUDA Graph replay;
- source, built-artifact and fresh-process Kernel Hub loading.

Performance publication additionally requires native-pointer parity and a
warmed eager/`torch.compile` comparison for the equivalent argmax+embedding
region. Sampling reports latency without a speedup against a different RNG.
