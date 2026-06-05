#!/usr/bin/env python3
"""Verify representative FlashRT Hub kernels under torch.compile."""

from __future__ import annotations

import argparse
from typing import Any, Callable

import torch
from kernels import get_kernel


REPOS = {
    "gemm": "flashrt/flashrt-gemm-epilogues",
    "fp8": "flashrt/flashrt-fp8-ffn",
    "vla": "flashrt/flashrt-vla-video",
    "nvfp4": "flashrt/flashrt-nvfp4",
    "smallm": "flashrt/flashrt-smallm-gemm",
    "fused": "flashrt/flashrt-fused-quant",
}


def assert_same(got: Any, expected: Any) -> None:
    if isinstance(got, tuple):
        if not isinstance(expected, tuple) or len(got) != len(expected):
            raise AssertionError("tuple output structure mismatch")
        for lhs, rhs in zip(got, expected):
            assert_same(lhs, rhs)
        return
    lhs = got.float() if got.is_floating_point() else got
    rhs = expected.float() if expected.is_floating_point() else expected
    torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)


def compile_check(name: str, fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
    eager = fn(*args)
    compiled = torch.compile(fn, fullgraph=True, mode="reduce-overhead")
    got = compiled(*args)
    assert_same(got, eager)
    print(f"PASS compile {name}")


def run(version: int) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not hasattr(torch, "float8_e4m3fn"):
        raise SystemExit("torch.float8_e4m3fn is required")

    print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    torch.manual_seed(321)
    dev = "cuda"

    gemm = get_kernel(REPOS["gemm"], version=version, trust_remote_code=True)
    a = torch.randn((16, 32), device=dev, dtype=torch.bfloat16).contiguous()
    b = torch.randn((32, 64), device=dev, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((64,), device=dev, dtype=torch.bfloat16).contiguous()
    compile_check("gemm_epilogues.bf16_gemm_bias_gelu", gemm.bf16_gemm_bias_gelu, (a, b, bias))

    fp8 = get_kernel(REPOS["fp8"], version=version, trust_remote_code=True)
    xs = torch.tensor([0.05], device=dev, dtype=torch.float32)
    ws = torch.tensor([0.04], device=dev, dtype=torch.float32)
    hs = torch.tensor([0.25], device=dev, dtype=torch.float32)
    inp = torch.clamp(torch.randn((8, 128), device=dev, dtype=torch.bfloat16).float() / xs, -448, 448).to(torch.float8_e4m3fn)
    up = torch.clamp(torch.randn((256, 128), device=dev, dtype=torch.bfloat16).float() / ws, -448, 448).to(torch.float8_e4m3fn)
    down = torch.clamp(torch.randn((64, 256), device=dev, dtype=torch.bfloat16).float() / ws, -448, 448).to(torch.float8_e4m3fn)
    up_b = torch.randn((256,), device=dev, dtype=torch.bfloat16)
    down_b = torch.randn((64,), device=dev, dtype=torch.bfloat16)
    compile_check("fp8_ffn.fp8_gemm_bf16", fp8.fp8_gemm_bf16, (inp, up, xs, ws))
    compile_check(
        "fp8_ffn.fp8_gelu_mlp_bf16",
        fp8.fp8_gelu_mlp_bf16,
        (inp, up, up_b, down, down_b, xs, ws, hs, ws),
    )

    vla = get_kernel(REPOS["vla"], version=version, trust_remote_code=True)
    q = (torch.randn((2, 4, 128), device=dev, dtype=torch.bfloat16) * 0.2).contiguous()
    weight = (torch.randn((128,), device=dev, dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    cos = torch.randn((64,), device=dev, dtype=torch.bfloat16).contiguous()
    sin = torch.randn((64,), device=dev, dtype=torch.bfloat16).contiguous()
    compile_check("vla_video.q_norm_rope_bf16", vla.q_norm_rope_bf16, (q, weight, cos, sin))

    nvfp4 = get_kernel(REPOS["nvfp4"], version=version, trust_remote_code=True)
    scales = torch.randint(0, 256, (4, 64), device=dev, dtype=torch.uint8).contiguous()
    compile_check("nvfp4.nvfp4_sf_linear_to_swizzled", nvfp4.nvfp4_sf_linear_to_swizzled, (scales,))

    smallm = get_kernel(REPOS["smallm"], version=version, trust_remote_code=True)
    k_dim, n_dim = 4096, 8
    a_packed = torch.full((k_dim // 2,), 0x11, device=dev, dtype=torch.uint8)
    b_packed = torch.full((n_dim, k_dim // 2), 0x11, device=dev, dtype=torch.uint8)
    sfa = torch.zeros((((1 + 127) // 128) * (((k_dim // 16) + 3) // 4) * 512,), device=dev, dtype=torch.uint8)
    sfb = torch.zeros((((n_dim + 127) // 128) * (((k_dim // 16) + 3) // 4) * 512,), device=dev, dtype=torch.uint8)
    compile_check(
        "smallm.nvfp4_w4a4_decode_matvec_bf16out",
        smallm.nvfp4_w4a4_decode_matvec_bf16out,
        (a_packed, b_packed, sfa, sfb),
    )

    fused = get_kernel(REPOS["fused"], version=version, trust_remote_code=True)
    gate = (torch.randn((3, 64), device=dev, dtype=torch.bfloat16) * 0.5).contiguous()
    up2 = (torch.randn((3, 64), device=dev, dtype=torch.bfloat16) * 0.5).contiguous()
    compile_check(
        "fused_quant.silu_mul_quant_nvfp4_swizzled_bf16",
        fused.silu_mul_quant_nvfp4_swizzled_bf16,
        (gate, up2),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    run(args.version)


if __name__ == "__main__":
    main()
