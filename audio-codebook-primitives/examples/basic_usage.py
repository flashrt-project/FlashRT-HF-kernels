import torch
from kernels import get_kernel

ops = get_kernel("flashrt/audio-codebook-primitives", version=1, trust_remote_code=True)
logits = torch.randn((8, 1026), device="cuda", dtype=torch.bfloat16)
codebook = torch.randn((8, 1026, 1024), device="cuda", dtype=torch.bfloat16)
codes, embedding = ops.delayed_codebook_argmax_embed_bf16(
    logits, codebook, delay=7, boc=1024
)
print(codes.shape, embedding.shape)
