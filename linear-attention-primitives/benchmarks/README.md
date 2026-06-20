# Benchmarks

`benchmark.py` compares the source or installed artifact against PyTorch eager
references on the same GPU.

Run:

```bash
python benchmarks/benchmark.py --backend source --warmup 100 --iters 1000
```
