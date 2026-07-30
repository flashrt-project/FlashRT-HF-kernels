# Tests

Run source validation:

```bash
python tests/test_padded_fp8_producers.py --backend source --mode full
```

Pass `--backend installed --artifact <variant-dir>` after kernel-builder output
is available.
