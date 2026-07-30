# Validation

Before release:

```bash
python tests/test_blockwise_fp8_producers.py --backend source --mode full
python benchmarks/benchmark.py --backend source --mode full
python ../scripts/prebuild_check.py \
  --package blockwise-fp8-producers \
  --check-config
```

Internal native parity:

```bash
python ../internal-tests/blockwise-fp8-producers/native_parity_benchmark.py
```

After HF Jobs:

```bash
python tests/test_blockwise_fp8_producers.py \
  --backend installed \
  --mode full \
  --artifact /path/to/built/variant
```

The release gate requires:

- all source and installed-artifact correctness rows pass;
- all original-native vs Hub outputs and scales are bitwise equal;
- fullgraph compile succeeds;
- artifact latency does not regress from package source or original native;
- unsupported widths and non-contiguous inputs fail before launch.
