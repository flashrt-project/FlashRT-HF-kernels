# Tests

Run installed-artifact validation with:

```bash
pytest -q tests/test_fa2_seqused_runtime.py
```

The matrix covers FP16/BF16, head dimensions 64/96/128/256, MHA/GQA,
causal BF16, partial tiles, aligned padded strides, device-resident sequence lengths,
split-KV, CUDA Graph replay, `torch.compile`, and unsupported-contract errors.
