# Validation

Validation target:

- Local GPU: RTX 5090 / Blackwell
- Source extension: `tests/test_fp8_kv_attention.py`
- Correctness metrics: max abs, mean abs, p99 abs, cosine similarity
- Reference: PyTorch BF16/FP8-dequant attention with the same XQA speculative
  mask and `1 / sqrt(256)` QK scaling

Required gate before publishing:

```bash
python fp8-kv-attention/tests/test_fp8_kv_attention.py --backend source --mode full
```

The package must pass all rows with:

- `max_abs <= 0.02`
- `mean_abs <= 0.0025`
- `cosine >= 0.999`

Installed-artifact validation should be rerun in a Torch version supported by
the Kernel Hub artifact matrix.

## RTX 5090 Source Results

Command:

```bash
python fp8-kv-attention/tests/test_fp8_kv_attention.py \
  --backend source \
  --mode full \
  --json-out internal-tests/fp8-kv-attention-source-full.json
```

Rows:

| Shape | q_seq | kv_seq | max_abs | mean_abs | p99_abs | cosine | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_128 | 1 | 128 | 0.000122 | 0.000009 | 0.000061 | 0.99999756 | PASS |
| decode_1024 | 1 | 1024 | 0.000061 | 0.000005 | 0.000031 | 0.99999523 | PASS |
| verify4_1024 | 4 | 1024 | 0.000061 | 0.000005 | 0.000031 | 0.99999517 | PASS |
| verify8_4096 | 8 | 4096 | 0.000031 | 0.000003 | 0.000015 | 0.99999541 | PASS |
