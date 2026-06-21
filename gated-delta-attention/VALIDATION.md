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
accumulation and BF16 state/output casts. Split/gating helpers are checked
against exact PyTorch tensor formulas. `gdn_chunk_from_conv_smem_bf16` and the
WY pipeline are checked end-to-end against the same recurrent reference.

## RTX 5090 Source Results

Command:

```bash
python gated-delta-attention/tests/test_gated_delta_attention.py \
  --backend source \
  --mode full \
  --json-out internal-tests/gated-delta-attention-v3-source-full.json
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
| split_s4 | split | 1 | 4 | 48 | 0.000000 | 0.000000 | 0.000000 | 1.00000000 | PASS |
| gating_s4 | gating | 1 | 4 | 48 | 0.000000 | 0.000000 | 0.000000 | 1.00000000 | PASS |
| chunk_from_conv_s4 | chunk_from_conv | 1 | 4 | 48 | 0.000015 | 0.000000 | 0.000000 | 1.00000000 | PASS |
| wy_pipeline_s4 | wy_pipeline | 1 | 4 | 48 | 0.000031 | 0.000004 | 0.000015 | 0.99999440 | PASS |
| wy_pipeline_s65 | wy_pipeline | 1 | 65 | 48 | 0.000107 | 0.000009 | 0.000038 | 0.99996358 | PASS |
| wy_mma_fla_s64 | wy_mma_fla | 1 | 64 | 48 | 0.000122 | 0.000010 | 0.000044 | 0.99996173 | PASS |
| wy_mma_fla_s65 | wy_mma_fla | 1 | 65 | 48 | 0.000107 | 0.000010 | 0.000040 | 0.99996245 | PASS |
| wy_mma_fla_s128 | wy_mma_fla | 1 | 128 | 48 | 0.000122 | 0.000011 | 0.000046 | 0.99994701 | PASS |

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

Result: same full rows pass for the local generated artifact.

For v3 release artifacts, rerun the same command against the HF Jobs artifact
before updating the installed-artifact claim.
