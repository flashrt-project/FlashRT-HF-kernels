# Validation

Required source gate before publishing:

```bash
python causal-conv1d-state/tests/test_causal_conv1d_state.py --backend source --mode full
```

Correctness metrics:

- `max_abs`
- `mean_abs`
- `p99_abs`
- cosine similarity

Tolerance:

- `max_abs <= 0.00390625`
- `mean_abs <= 0.0008`
- `cosine >= 0.999`

The reference is a deterministic PyTorch BF16 contract with FP32 accumulation,
causal left padding, optional bias, optional SiLU, and the same state update
layout `(B, C, K - 1)`.

## RTX 5090 Source Results

Command:

```bash
python causal-conv1d-state/tests/test_causal_conv1d_state.py \
  --backend source \
  --mode full \
  --json-out internal-tests/causal-conv1d-state-source-full.json
```

Rows:

| Shape | Kind | B | S | C | K | max_abs | mean_abs | p99_abs | cosine | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fwd_s8_c1024 | fwd | 1 | 8 | 1024 | 4 | 0.000000 | 0.000000 | 0.000000 | 1.00000000 | PASS |
| decode_c1024 | decode | 1 | 1 | 1024 | 4 | 0.000000 | 0.000000 | 0.000000 | 0.99999988 | PASS |
| inout_c1024 | inout | 1 | 1 | 1024 | 4 | 0.000000 | 0.000000 | 0.000000 | 0.99999988 | PASS |
| chunk_s8_c1024 | chunk | 1 | 8 | 1024 | 4 | 0.000000 | 0.000000 | 0.000000 | 0.99999988 | PASS |
| parallel_s16_c1024 | parallel | 1 | 16 | 1024 | 4 | 0.000000 | 0.000000 | 0.000000 | 1.00000000 | PASS |
| gqa_s8_c10240 | gqa | 1 | 8 | 10240 | 4 | 0.000000 | 0.000000 | 0.000000 | 1.00000012 | PASS |

## Generated Artifact Smoke

Local generated pyproject wheel path:

```bash
DEBUG=0 MAX_JOBS=8 NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST=12.0a \
  python -m pip wheel . -w /tmp/causal-conv1d-wheel -v --no-build-isolation

python tests/test_causal_conv1d_state.py \
  --backend installed \
  --artifact build/lib.linux-x86_64-cpython-313 \
  --mode full \
  --json-out ../internal-tests/causal-conv1d-state-installed-local-full.json
```

Result: same 6/6 rows pass.
