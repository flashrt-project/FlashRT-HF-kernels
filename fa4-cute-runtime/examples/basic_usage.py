"""Minimal Kernel Hub and CUDA Graph usage."""

import torch
from kernels import get_kernel


fa4 = get_kernel("flashrt/fa4-cute-runtime", version=1)
q = torch.randn(1, 456, 8, 256, device="cuda", dtype=torch.float16)
k = torch.randn(1, 968, 1, 256, device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
out = torch.empty_like(q)
seqused_k = torch.tensor([456], device="cuda", dtype=torch.int32)

fa4.forward_static(q, k, v, out, causal=False, seqused_k=seqused_k)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    fa4.forward_static(q, k, v, out, causal=False, seqused_k=seqused_k)
graph.replay()
