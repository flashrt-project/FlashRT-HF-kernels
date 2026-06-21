from kernels import get_kernel
import torch

ops = get_kernel("flashrt/grouped-moe-gemv", version=1, trust_remote_code=True)
K, N = 256, 128
x = torch.ones((K,), device="cuda", dtype=torch.bfloat16)
w = torch.full((N, K // 2), 0x11, device="cuda", dtype=torch.uint8)
sfb = torch.full((512,), 0x38, device="cuda", dtype=torch.uint8)
y = ops.w4a16_decode_gemv_bf16(x, w, sfb)
print(y.shape)
