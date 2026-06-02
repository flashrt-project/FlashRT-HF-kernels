import torch

from flashrt_fused_quant import (
    silu_mul_merged_quant_nvfp4_swizzled_bf16,
    silu_mul_quant_nvfp4_swizzled_bf16,
)


def run_split_ffn_quant(tokens: int = 64, hidden: int = 12288):
    gate = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    up = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    return silu_mul_quant_nvfp4_swizzled_bf16(gate.contiguous(), up.contiguous())


def run_merged_ffn_quant(tokens: int = 64, hidden: int = 12288):
    gate = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    up = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    merged_gate_up = torch.cat([gate, up], dim=1).contiguous()
    return silu_mul_merged_quant_nvfp4_swizzled_bf16(merged_gate_up)


if __name__ == "__main__":
    packed, scales = run_split_ffn_quant()
    merged_packed, merged_scales = run_merged_ffn_quant()
    torch.cuda.synchronize()
    print("split packed", tuple(packed.shape), packed.dtype)
    print("split scales", tuple(scales.shape), scales.dtype)
    print("merged packed", tuple(merged_packed.shape), merged_packed.dtype)
    print("merged scales", tuple(merged_scales.shape), merged_scales.dtype)
