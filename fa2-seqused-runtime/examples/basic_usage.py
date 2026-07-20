import torch
from kernels import get_kernel


fa2 = get_kernel("flashrt/fa2-seqused-runtime", version=1)

q = torch.randn(1, 32, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 512, 2, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
out, lse = fa2.allocate_outputs(q)
seqused_k = torch.tensor([384], device="cuda", dtype=torch.int32)

# Allocation-free runtime call. All tensors can be captured by a CUDA Graph.
fa2.forward_seqused_static(
    q,
    k,
    v,
    seqused_k,
    out=out,
    softmax_lse=lse,
)
print(out.shape)
