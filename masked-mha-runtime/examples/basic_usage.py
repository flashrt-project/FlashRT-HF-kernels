import torch
from kernels import get_kernel

mha = get_kernel("flashrt/masked-mha-runtime", version=1)
q = torch.randn((41, 32, 48), device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
logits = mha.allocate_workspace(q, k)
out = torch.empty_like(q)
mha.forward_static(q, k, v, logits=logits, out=out)
