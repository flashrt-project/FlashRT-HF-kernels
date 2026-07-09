# Examples

Source-tree smoke:

```bash
PYTHONPATH=flashrt-flex-attention-train/torch-ext python - <<'PY'
import torch
import flashrt_flex_attention_train as flex_ops

B, H, P, A, D = 2, 8, 16, 10, 256
q = torch.randn(B, H, P + A, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

out = flex_ops.flex_attention(q, k, v, prefix_len=P, action_block_size=5)
out.float().square().mean().backward()
print(out.shape)
PY
```
