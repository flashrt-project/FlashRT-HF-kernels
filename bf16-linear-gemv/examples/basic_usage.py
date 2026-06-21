from kernels import get_kernel
import torch

ops = get_kernel("flashrt/bf16-linear-gemv", version=1, trust_remote_code=True)
x = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)
w = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16)
y = ops.bf16_decode_gemv_unrolled_bf16(x, w)
print(y.shape)
