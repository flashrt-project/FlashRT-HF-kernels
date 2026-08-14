# Benchmark

Run source or installed-artifact timing with preallocated workspaces:

```bash
python sageattention3-blackwell/benchmarks/benchmark.py --backend source --mode full
python sageattention3-blackwell/benchmarks/benchmark.py \
  --backend installed --artifact <build/variant> --mode full
```

The Sage2 and Sage3 columns both include Q/K/V quantization and attention with
preallocated workspaces. PyTorch SDPA uses the same BF16 tensors. Allocation
and Sage3 centering/delta preparation are outside all timed regions.
