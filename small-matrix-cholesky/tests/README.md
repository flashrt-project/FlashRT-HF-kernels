# Tests

Source smoke test:

```bash
python tests/test_small_matrix_cholesky.py \
  --backend source --mode smoke \
  --registration-include /path/to/kernel-builder/templates/torch
```

Use `--mode full` to add the GPU MODE batch sizes and leading-batch-dimension
coverage. Both modes cover tail batches for every matrices-per-block schedule,
CUDA Graph replay, and `torch.compile(fullgraph=True)`. If two CUDA devices are
visible, both modes also validate an `n=128` launch on the non-default device.

Installed artifact:

```bash
python tests/test_small_matrix_cholesky.py \
  --backend installed --mode full --artifact /path/to/artifact
```
