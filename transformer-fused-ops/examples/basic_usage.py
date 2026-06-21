from kernels import get_kernel
import torch

ops = get_kernel("flashrt/transformer-fused-ops", version=1, trust_remote_code=True)
x = torch.randn((8, 128), device="cuda", dtype=torch.bfloat16)
gate = torch.randn_like(x)
weight = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
y = ops.rms_norm_gated_silu_bf16(x, gate, weight)
print(y.shape)
