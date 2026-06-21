from kernels import get_kernel
import torch

ops = get_kernel("flashrt/linear-attention-seq-state", version=1, trust_remote_code=True)
q = torch.randn((4, 2, 128), device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
g = torch.zeros((4, 2), device="cuda", dtype=torch.bfloat16)
beta = torch.ones_like(g)
state = torch.zeros((2, 128, 128), device="cuda", dtype=torch.bfloat16)
out, state = ops.gated_delta_recurrent_seq_bf16(q, k, v, g, beta, state)
print(out.shape)
