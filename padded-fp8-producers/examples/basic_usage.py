import torch
from kernels import get_kernel

k = get_kernel(
    "flashrt/padded-fp8-producers", version=1, trust_remote_code=True
)
x = torch.randn(1, 51, 2048, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(2048, device="cuda", dtype=torch.bfloat16)
gamma = torch.zeros(1, 2048, device="cuda", dtype=torch.bfloat16)
beta = torch.zeros_like(gamma)
scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)

output = k.adaptive_rms_norm_quant_fp8_padded_bf16(
    x, weight, gamma, beta, scale, padded_rows=64
)
print(output.shape, output.dtype)
