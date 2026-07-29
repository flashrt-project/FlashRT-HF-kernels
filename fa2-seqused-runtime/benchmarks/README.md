# Benchmarks

```bash
python benchmarks/benchmark.py --dtype bf16
python benchmarks/benchmark.py --dtype fp16
```

The FlashRT timing includes split-KV dispatch when the package heuristic selects
it. The SDPA baseline explicitly materializes repeated K/V heads for GQA, so
results must label that fact. Installed-artifact results are recorded only after
the corresponding Hub build is available. The matrix includes GROOT DiT D=48,
SigLIP D=72, Qwen/GQA D=128, and generic decode/prefill rows.
