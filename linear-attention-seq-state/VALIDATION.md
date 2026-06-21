# Validation

Required source gate:

```bash
python linear-attention-seq-state/tests/test_linear_attention_seq_state.py --backend source --mode full
```

Reference: deterministic PyTorch eager sequential scan with the same BF16
state writeback contract.
