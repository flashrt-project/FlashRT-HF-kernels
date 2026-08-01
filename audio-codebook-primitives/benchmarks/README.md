# Benchmarks

The public benchmark compares fused argmax+embedding against equivalent
warmed PyTorch eager and `torch.compile`. Sampling latency is reported without
a speedup claim because its counter-based RNG is not `torch.multinomial`.

Built-artifact results are published only after native-pointer parity passes.
