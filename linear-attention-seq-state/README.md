# linear-attention-seq-state

FlashRT native CUDA sequential state-scan kernels for Gated DeltaNet style
linear attention prefill.

## Functions

- `gated_delta_recurrent_seq_bf16(q, k, v, g, beta, state, out=None, use_qk_l2norm=False)`

The API scans `(S,H,128)` in one launch and updates `state (H,128,128)` in
place. It is meant for prefill paths that would otherwise launch a recurrent
kernel once per token.

## Validation

```bash
python linear-attention-seq-state/tests/test_linear_attention_seq_state.py --backend source --mode full
```
