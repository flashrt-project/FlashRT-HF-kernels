"""Minimal Kernel Hub and CUDA Graph usage."""

import torch
from kernels import get_kernel


fa4 = get_kernel("flashrt/fa4-cute-runtime", version=1)
q = torch.randn(1, 277, 16, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 277, 8, 128, device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
out = torch.empty_like(q)

fa4.forward_static(q, k, v, out, causal=True)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    fa4.forward_static(q, k, v, out, causal=True)
graph.replay()
