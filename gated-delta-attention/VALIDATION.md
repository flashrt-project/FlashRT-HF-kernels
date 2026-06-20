# Validation

Required source gate before publishing:

```bash
python gated-delta-attention/tests/test_gated_delta_attention.py --backend source --mode full
```

Correctness metrics:

- `max_abs`
- `mean_abs`
- `p99_abs`
- cosine similarity

The reference uses the same recurrent Gated DeltaNet math with FP32 internal
accumulation and BF16 state/output casts.

## RTX 5090 Source Results

Command:

```bash
python gated-delta-attention/tests/test_gated_delta_attention.py \
  --backend source \
  --mode full \
  --json-out internal-tests/gated-delta-attention-source-full.json
```

Rows:

| Shape | Kind | B | S | H | max_abs | mean_abs | p99_abs | cosine | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| recurrent_h4 | recurrent | 1 | 1 | 4 | 0.000000 | 0.000000 | 0.000000 | 0.99999988 | PASS |
| inout_h4 | inout | 1 | 1 | 4 | 0.000000 | 0.000000 | 0.000000 | 0.99999988 | PASS |
| f32state_h4 | f32state | 1 | 1 | 4 | 0.000000 | 0.000000 | 0.000000 | 1.00000000 | PASS |
| chunk_s4_h4 | chunk | 1 | 4 | 4 | 0.000000 | 0.000000 | 0.000000 | 0.99999994 | PASS |
| chunk_smem_s4_h4 | chunk_smem | 1 | 4 | 4 | 0.000000 | 0.000000 | 0.000000 | 0.99999994 | PASS |
| recurrent_h48 | recurrent | 1 | 1 | 48 | 0.000002 | 0.000000 | 0.000000 | 1.00000000 | PASS |

## Generated Artifact Smoke

Local generated pyproject wheel path:

```bash
DEBUG=0 MAX_JOBS=8 NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST=12.0a \
  python -m pip wheel . -w /tmp/gated-delta-wheel -v --no-build-isolation

python tests/test_gated_delta_attention.py \
  --backend installed \
  --artifact build/lib.linux-x86_64-cpython-313 \
  --mode full \
  --json-out ../internal-tests/gated-delta-attention-installed-local-full.json
```

Result: same 6/6 rows pass.
