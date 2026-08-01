# Results

## RTX 5090 source gate

Environment: NVIDIA GeForce RTX 5090, PyTorch 2.9.1+cu128. Higgs workload:
`num_codebooks=8`, `vocab=1026`, `hidden=1024`, delayed argmax plus embedding
sum. CUDA events, 50 warm-up and 500 measured iterations.

| Path | Latency (us) |
| --- | ---: |
| PyTorch eager | 21.708 |
| `torch.compile(fullgraph=True)` | 29.831 |
| FlashRT tensor wrapper | 6.089 |
| Raw native op | 6.093 |

Wrapper and raw-op latency are within measurement noise, so the tensor API
does not add a measurable hot-path penalty. Exact codes and BF16 embedding are
gated before timing. These are source-extension results; installed-artifact
results are added only after HF Jobs build and fresh-process `get_kernel` tests.
