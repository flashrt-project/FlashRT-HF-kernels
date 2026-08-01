import torch
from kernels import get_kernel

ops = get_kernel("flashrt/fused-mlp-megakernels-blackwell", version=1)
x = torch.randn(768, 2048, device="cuda", dtype=torch.float16)
gate_weight = torch.randn(8192, 2048, device="cuda", dtype=torch.float16)
up_weight = torch.randn_like(gate_weight)
output = ops.fp16_geglu_fused(x, gate_weight, up_weight)
print(output.shape, output.dtype)
