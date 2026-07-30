import torch
from kernels import get_kernel

ops = get_kernel(
    "flashrt/blockwise-fp8-producers",
    version=1,
    trust_remote_code=True,
)

x = torch.randn((51, 4096), device="cuda", dtype=torch.bfloat16)
weight = torch.ones((4096,), device="cuda", dtype=torch.bfloat16)
fp8_x, block_scale = ops.rms_norm_fp8_block128_bf16(x, weight)
print(fp8_x.shape, fp8_x.dtype, block_scale.shape, block_scale.dtype)
