# Benchmarks

`benchmark.py` compares the package API against the current two-call SDPA
baseline at PI052 FlexAttention replacement shapes. It records forward time, forward+backward time,
peak memory, forward max absolute diff, and gradient norm relative diff.

The default shape is the measured PI052 flow-only shape:

- `B=4`
- `heads=8`
- `head_dim=256`
- `prefix_len=700`
- `action_blocks=5`
- `action_block_size=50`

For hardware/tile sweeps, run one process per shape and preset to avoid
`torch.compile` guard reuse noise:

```bash
python flashrt-flex-attention-train/benchmarks/shape_matrix.py --presets a100 --output /tmp/a100_flex_matrix.jsonl
python flashrt-flex-attention-train/benchmarks/shape_matrix.py --presets consumer --output /tmp/consumer_flex_matrix.jsonl
```
