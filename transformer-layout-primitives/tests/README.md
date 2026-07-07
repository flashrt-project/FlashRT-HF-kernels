# Tests

Run source correctness:

```bash
python tests/test_transformer_layout_primitives.py --backend source --mode full
```

Run installed-artifact correctness after publishing:

```bash
python tests/test_transformer_layout_primitives.py --backend installed --mode full
```
