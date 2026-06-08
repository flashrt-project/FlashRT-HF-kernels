# Source Benchmark Results

Environment: NVIDIA GeForce RTX 5090 local source-extension build.
Baseline: PyTorch eager tensor postprocess chain with matching BF16 math.

| Shape | V,A,U rows | Dim | FlashRT us | Eager us | vs eager | p99 abs | Cosine | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| small | 64,8,4 | 1024 | 3.178 | 61.074 | 19.22x | 0.000000 | 0.99999994 | PASS |
| vla_2k | 2520,16,16 | 3072 | 10.326 | 187.622 | 18.17x | 0.000000 | 1.00000000 | PASS |
| vla_4k | 4096,16,16 | 3072 | 20.352 | 361.285 | 17.75x | 0.000000 | 1.00000012 | PASS |
