#!/usr/bin/env python3
"""Correctness tests for grouped-moe-gemv."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "grouped-moe-gemv"
REGISTRATION_INCLUDE = ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    def w4a16_decode_gemv_bf16(self, x, w, sfb, alpha=1.0):
        out = torch.empty((w.shape[0],), device=x.device, dtype=torch.bfloat16)
        self.ops.w4a16_decode_gemv_bf16(x, w, sfb, float(alpha), out)
        return out

    def grouped_w4a16_gemv_bf16(self, acts, w_stack, sfb_stack, alpha_stack, expert_idx, w_stride, sfb_stride, n, out=None):
        if out is None:
            out = torch.empty((acts.shape[0], n), device=acts.device, dtype=torch.bfloat16)
        self.ops.grouped_w4a16_gemv_bf16(acts, w_stack, sfb_stack, alpha_stack, expert_idx, int(w_stride), int(sfb_stride), out)
        return out

    def quantize_activations_nvfp4_bf16(self, x, packed=None, sfa=None):
        if packed is None:
            packed = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=torch.uint8)
        if sfa is None:
            sfa = torch.empty((sfb_bytes(x.shape[0], x.shape[1]),), device=x.device, dtype=torch.uint8)
        self.ops.quantize_activations_nvfp4_bf16(x, packed, sfa)
        return packed, sfa

    def quantize_weights_nvfp4_bf16(self, w, packed=None, sfb=None):
        if packed is None:
            packed = torch.empty((w.shape[0], w.shape[1] // 2), device=w.device, dtype=torch.uint8)
        if sfb is None:
            sfb = torch.empty((sfb_bytes(w.shape[0], w.shape[1]),), device=w.device, dtype=torch.uint8)
        self.ops.quantize_weights_nvfp4_bf16(w, packed, sfb)
        return packed, sfb

    def grouped_w4a4_gemv_bf16(self, a, w, sfa, sfb, alpha, idx, out=None):
        if out is None:
            out = torch.empty((a.shape[0], idx.shape[1], w.shape[1]), device=a.device, dtype=torch.bfloat16)
        self.ops.grouped_w4a4_gemv_bf16(a, w, sfa, sfb, alpha, idx, out)
        return out


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major == 11 and minor == 0:
        return "11.0a"
    return "12.0a" if major >= 12 else f"{major}.{minor}"


def _is_sm110() -> bool:
    return torch.cuda.get_device_capability(0) == (11, 0)


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "grouped_moe_gemv_source_test"
    if _is_sm110():
        kernel_sources = [
            str(PACKAGE / "csrc" / "kernels" / "w4a16_edge_sm120.cu"),
            str(PACKAGE / "csrc" / "kernels" / "sm110_dispatch.cu"),
            str(PACKAGE / "csrc" / "kernels" / "quantize_activations_nvfp4.cu"),
        ]
        cuda_flags = [
            "-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL",
            "-DFLASHRT_W4A16_EDGE_UNROLL=2",
        ]
    else:
        kernel_sources = [
            str(PACKAGE / "csrc" / "kernels" / "nexn2_w4a16_gemv.cu"),
            str(PACKAGE / "csrc" / "kernels" / "nexn2_moe_grouped_w4a16.cu"),
            str(PACKAGE / "csrc" / "kernels" / "grouped_w4a4_gemv_sm120.cu"),
            str(PACKAGE / "csrc" / "kernels" / "quantize_activations_nvfp4.cu"),
        ]
        cuda_flags = ["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"]
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            *kernel_sources,
        ],
        extra_include_paths=[
            str(PACKAGE / "csrc"),
            str(REGISTRATION_INCLUDE),
            os.environ.get(
                "CUTLASS_INCLUDE",
                "/home/heima/suliang/PI/official/FlashRT/third_party/cutlass/include",
            ),
        ],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=cuda_flags,
        is_python_module=False,
        verbose=False,
    )
    torch.library.register_fake(f"{namespace}::quantize_activations_nvfp4_bf16")(
        lambda activations, packed, sfa: None
    )
    torch.library.register_fake(f"{namespace}::quantize_weights_nvfp4_bf16")(
        lambda weights, packed, sfb: None
    )
    torch.library.register_fake(f"{namespace}::grouped_w4a4_gemv_bf16")(
        lambda activations_packed, weight_stack, sfa, sfb_stack, alpha_stack,
        expert_idx, out: None
    )
    torch.library.register_fake(f"{namespace}::w4a16_decode_gemv_bf16")(
        lambda x, weight, sfb, alpha, out: None
    )
    torch.library.register_fake(f"{namespace}::grouped_w4a16_gemv_bf16")(
        lambda activations, weight_stack, sfb_stack, alpha_stack, expert_idx,
        w_stride, sfb_stride, out: None
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("grouped_moe_gemv")
    finally:
        if artifact:
            sys.path.remove(artifact)


def sfb_bytes(rows: int, k: int) -> int:
    n_blocks = k // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 512


def assert_constant(name: str, got: torch.Tensor, expected: float) -> None:
    ref = torch.full_like(got.float(), expected)
    diff = (got.float() - ref).abs()
    max_abs = float(diff.max().item())
    if max_abs > 0.0:
        raise AssertionError(f"{name}: expected exact {expected}, max_abs={max_abs}")


def _metrics(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    got_f = got.float().reshape(-1)
    ref_f = ref.float().reshape(-1)
    error = (got_f - ref_f).abs()
    return {
        "max_abs": float(error.max().item()),
        "p99_abs": float(torch.quantile(error, 0.99).item()),
        "mean_abs": float(error.mean().item()),
        "cosine": float(torch.nn.functional.cosine_similarity(got_f, ref_f, dim=0).item()),
    }


def _assert_kernel_metrics(name: str, metrics: dict[str, float]) -> None:
    if metrics["cosine"] < 0.99999 or metrics["p99_abs"] > 0.03125:
        raise AssertionError(f"{name}: correctness metrics failed: {metrics}")


def _decode_ue4m3(raw: torch.Tensor) -> torch.Tensor:
    value = raw.to(torch.int32)
    exponent = (value >> 3) & 0xF
    mantissa = value & 0x7
    subnormal = mantissa.float() * (2.0**-9)
    normal = (1.0 + mantissa.float() / 8.0) * torch.pow(
        torch.tensor(2.0, device=raw.device), exponent.float() - 7.0
    )
    return torch.where(exponent == 0, subnormal, normal)


def _dequantize_nvfp4(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    rows, k_half = packed.shape
    k = k_half * 2
    fp4_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=packed.device,
    )
    values = torch.stack(
        (fp4_lut[(packed & 0xF).long()], fp4_lut[(packed >> 4).long()]), dim=-1
    ).reshape(rows, k)
    blocks = k // 16
    row = torch.arange(rows, device=packed.device, dtype=torch.long)[:, None]
    block = torch.arange(blocks, device=packed.device, dtype=torch.long)[None, :]
    n_col_super = (blocks + 3) // 4
    offsets = (
        ((row >> 7) * n_col_super + (block >> 2)) * 512
        + (row & 31) * 16
        + ((row >> 5) & 3) * 4
        + (block & 3)
    )
    scales = _decode_ue4m3(sf[offsets])
    return values * scales.repeat_interleave(16, dim=1)


def _w4a4_constant_case(ops, m: int, top_k: int, n: int, k: int) -> int:
    experts = max(top_k, 8)
    a = torch.full((m, k // 2), 0x11, device="cuda", dtype=torch.uint8)
    w = torch.full((experts, n, k // 2), 0x11, device="cuda", dtype=torch.uint8)
    sfa = torch.full((sfb_bytes(m, k),), 0x38, device="cuda", dtype=torch.uint8)
    sf_one = sfb_bytes(n, k)
    sfb = torch.full((experts, sf_one), 0x38, device="cuda", dtype=torch.uint8)
    alpha = torch.linspace(0.5, 1.5, experts, device="cuda", dtype=torch.float32)
    idx = (torch.arange(m * top_k, device="cuda", dtype=torch.int32).reshape(m, top_k) % experts).contiguous()
    got = ops.grouped_w4a4_gemv_bf16(a, w, sfa, sfb, alpha, idx)
    torch.cuda.synchronize()
    expected = (k * 0.25 * alpha[idx.long()]).unsqueeze(-1).expand_as(got)
    if not torch.equal(got, expected.to(torch.bfloat16)):
        raise AssertionError(
            f"W4A4 constant mismatch M={m} top_k={top_k} N={n} K={k}: "
            f"{_metrics(got, expected)}"
        )
    return 1


def _w4a4_random_case(ops, m: int, top_k: int, n: int, k: int) -> int:
    torch.manual_seed(1000 + m + n + k)
    experts = max(8, top_k)
    x = (torch.randn((m, k), device="cuda") * 0.25).to(torch.bfloat16)
    weights = (torch.randn((experts, n, k), device="cuda") * 0.08).to(torch.bfloat16)
    a, sfa = ops.quantize_activations_nvfp4_bf16(x)
    w = torch.empty((experts, n, k // 2), device="cuda", dtype=torch.uint8)
    sf_one = sfb_bytes(n, k)
    sfb = torch.empty((experts, sf_one), device="cuda", dtype=torch.uint8)
    for expert in range(experts):
        ops.quantize_weights_nvfp4_bf16(
            weights[expert], packed=w[expert], sfb=sfb[expert]
        )
    alpha = torch.linspace(0.75, 1.25, experts, device="cuda", dtype=torch.float32)
    idx = torch.randint(experts, (m, top_k), device="cuda", dtype=torch.int32)
    got = ops.grouped_w4a4_gemv_bf16(a, w, sfa, sfb, alpha, idx)

    # Native loop parity: same packed values and scale layouts, but one top-k
    # column per launch. This catches routed-pair indexing and SFA-row errors
    # without conflating them with expected NVFP4 quantization loss.
    loop = torch.empty_like(got)
    for route in range(top_k):
        one = ops.grouped_w4a4_gemv_bf16(
            a, w, sfa, sfb, alpha, idx[:, route : route + 1].contiguous()
        )
        loop[:, route : route + 1].copy_(one)
    torch.cuda.synchronize()
    if not torch.equal(got, loop):
        raise AssertionError(
            f"grouped-vs-native-loop mismatch M={m} top_k={top_k} N={n} K={k}: "
            f"{_metrics(got, loop)}"
        )

    x_deq = _dequantize_nvfp4(a, sfa)
    w_deq = torch.stack(
        [_dequantize_nvfp4(w[e], sfb[e]) for e in range(experts)], dim=0
    )
    selected_deq = w_deq[idx.long()]
    contract_ref = torch.einsum("mk,mtnk->mtn", x_deq, selected_deq)
    contract_ref = contract_ref * alpha[idx.long()].unsqueeze(-1)
    contract_metrics = _metrics(got, contract_ref)
    _assert_kernel_metrics(
        f"W4A4 contract M={m} top_k={top_k} N={n} K={k}", contract_metrics
    )

    selected_source = weights[idx.long()]
    source_ref = torch.einsum("mk,mtnk->mtn", x.float(), selected_source.float())
    source_ref = source_ref * alpha[idx.long()].unsqueeze(-1)
    quant_metrics = _metrics(got, source_ref)
    if quant_metrics["cosine"] < 0.985:
        raise AssertionError(f"NVFP4 quality gate failed: {quant_metrics}")
    print(
        f"W4A4 random M={m} top_k={top_k} N={n} K={k}: "
        f"contract={contract_metrics} quant_vs_bf16={quant_metrics}"
    )
    return 2


def _cuda_graph_case(ops) -> int:
    m, top_k, experts, n, k = 7, 8, 8, 128, 512
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((experts, n, k), device="cuda", dtype=torch.bfloat16) * 0.05
    a = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
    sfa = torch.empty((sfb_bytes(m, k),), device="cuda", dtype=torch.uint8)
    w = torch.empty((experts, n, k // 2), device="cuda", dtype=torch.uint8)
    sfb = torch.empty((experts, sfb_bytes(n, k)), device="cuda", dtype=torch.uint8)
    for expert in range(experts):
        ops.quantize_weights_nvfp4_bf16(
            weights[expert], packed=w[expert], sfb=sfb[expert]
        )
    alpha = torch.ones((experts,), device="cuda", dtype=torch.float32)
    idx = torch.arange(m * top_k, device="cuda", dtype=torch.int32).reshape(m, top_k) % experts
    out = torch.empty((m, top_k, n), device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        ops.quantize_activations_nvfp4_bf16(x, packed=a, sfa=sfa)
        ops.grouped_w4a4_gemv_bf16(a, w, sfa, sfb, alpha, idx, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.quantize_activations_nvfp4_bf16(x, packed=a, sfa=sfa)
        ops.grouped_w4a4_gemv_bf16(a, w, sfa, sfb, alpha, idx, out=out)
    graph.replay()
    torch.cuda.synchronize()
    first = out.clone()
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(first, out):
        raise AssertionError("CUDA Graph replay was not bit-identical")
    idx.copy_(torch.flip(idx, dims=[1]))
    graph.replay()
    torch.cuda.synchronize()
    replay = out.clone()
    eager = ops.grouped_w4a4_gemv_bf16(a, w, sfa, sfb, alpha, idx)
    torch.cuda.synchronize()
    if not torch.equal(replay, eager):
        raise AssertionError("CUDA Graph did not reread device-side expert_idx")
    return 2


def _torch_compile_case(ops) -> int:
    m, top_k, experts, n, k = 2, 2, 4, 64, 128
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    a = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
    sfa = torch.empty((sfb_bytes(m, k),), device="cuda", dtype=torch.uint8)
    w = torch.full((experts, n, k // 2), 0x11, device="cuda", dtype=torch.uint8)
    sfb = torch.full(
        (experts, sfb_bytes(n, k)), 0x38, device="cuda", dtype=torch.uint8
    )
    alpha = torch.ones((experts,), device="cuda", dtype=torch.float32)
    idx = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    out = torch.empty((m, top_k, n), device="cuda", dtype=torch.bfloat16)

    def region(x_arg, idx_arg, packed_arg, sfa_arg, out_arg):
        ops.quantize_activations_nvfp4_bf16(
            x_arg, packed=packed_arg, sfa=sfa_arg
        )
        return ops.grouped_w4a4_gemv_bf16(
            packed_arg, w, sfa_arg, sfb, alpha, idx_arg, out=out_arg
        )

    compiled = torch.compile(region, fullgraph=True)
    got = compiled(x, idx, a, sfa, out)
    torch.cuda.synchronize()
    compiled_copy = got.clone()
    eager = region(x, idx, a, sfa, out)
    torch.cuda.synchronize()
    if not torch.equal(compiled_copy, eager):
        raise AssertionError(f"torch.compile mismatch: {_metrics(compiled_copy, eager)}")
    return 1


def _sm110_w4a4_rejection_case(ops) -> int:
    a = torch.empty((1, 64), device="cuda", dtype=torch.uint8)
    w = torch.empty((1, 64, 64), device="cuda", dtype=torch.uint8)
    sfa = torch.empty((sfb_bytes(1, 128),), device="cuda", dtype=torch.uint8)
    sfb = torch.empty((1, sfb_bytes(64, 128)), device="cuda", dtype=torch.uint8)
    alpha = torch.ones((1,), device="cuda", dtype=torch.float32)
    idx = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    try:
        ops.grouped_w4a4_gemv_bf16(a, w, sfa, sfb, alpha, idx)
    except RuntimeError as exc:
        if "requires SM120/SM121" not in str(exc):
            raise
    else:
        raise AssertionError("SM110 must reject the SM120-only W4A4 path")
    return 1


def _sm110_graph_case(ops) -> int:
    slots, experts, n, k = 8, 4, 256, 512
    acts = torch.randn((slots, k), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((experts, n, k), device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((experts, n, k // 2), device="cuda", dtype=torch.uint8)
    sf_one = sfb_bytes(n, k)
    scales = torch.empty((experts, sf_one), device="cuda", dtype=torch.uint8)
    for expert in range(experts):
        ops.quantize_weights_nvfp4_bf16(
            weights[expert], packed=packed[expert], sfb=scales[expert]
        )
    alpha = torch.ones((experts,), device="cuda", dtype=torch.float32)
    idx = torch.arange(slots, device="cuda", dtype=torch.int32) % experts
    out = torch.empty((slots, n), device="cuda", dtype=torch.bfloat16)
    args = (acts, packed, scales, alpha, idx, n * k // 2, sf_one, n)
    for _ in range(3):
        ops.grouped_w4a16_gemv_bf16(*args, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.grouped_w4a16_gemv_bf16(*args, out=out)
    graph.replay()
    torch.cuda.synchronize()
    first = out.clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, first, rtol=0, atol=0)
    idx.copy_(torch.flip(idx, dims=[0]))
    graph.replay()
    torch.cuda.synchronize()
    replay = out.clone()
    eager = ops.grouped_w4a16_gemv_bf16(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(replay, eager, rtol=0, atol=0)
    return 2


def run(ops, mode: str) -> int:
    shapes = (
        [(64, 128), (128, 256)]
        if mode == "smoke"
        else [
            (64, 128),
            (128, 256),
            (256, 512),
            (1024, 2048),  # gate/up decode profile
            (2048, 512),   # down decode/verify profile
        ]
    )
    count = 0
    for n, k in shapes:
        x = torch.ones((k,), device="cuda", dtype=torch.bfloat16)
        packed = torch.full((n, k // 2), 0x11, device="cuda", dtype=torch.uint8)
        sfb = torch.full((sfb_bytes(n, k),), 0x38, device="cuda", dtype=torch.uint8)
        got = ops.w4a16_decode_gemv_bf16(x, packed, sfb, alpha=1.0)
        torch.cuda.synchronize()
        assert_constant(f"w4a16_decode n={n} k={k}", got, k * 0.5)
        count += 1

        slots = 4
        experts = 3
        acts = torch.ones((slots, k), device="cuda", dtype=torch.bfloat16)
        w_stack = torch.full((experts, n, k // 2), 0x11, device="cuda", dtype=torch.uint8).contiguous()
        sfb_one = sfb_bytes(n, k)
        sfb_stack = torch.full((experts, sfb_one), 0x38, device="cuda", dtype=torch.uint8).contiguous()
        alpha = torch.tensor([1.0, 0.5, 2.0], device="cuda", dtype=torch.float32)
        expert_idx = torch.tensor([0, 1, 2, 1], device="cuda", dtype=torch.int32)
        got_g = ops.grouped_w4a16_gemv_bf16(
            acts, w_stack, sfb_stack, alpha, expert_idx,
            w_stride=n * k // 2, sfb_stride=sfb_one, n=n)
        torch.cuda.synchronize()
        expected = torch.tensor([k * 0.5, k * 0.25, k * 1.0, k * 0.25], device="cuda", dtype=torch.float32)[:, None]
        diff = (got_g.float() - expected).abs()
        if float(diff.max().item()) > 0.0:
            raise AssertionError("grouped_w4a16_gemv_bf16 constant mismatch")
        count += 1
    if _is_sm110():
        count += _sm110_w4a4_rejection_case(ops)
        count += _sm110_graph_case(ops)
        return count

    w4a4_shapes = [(1, 8, 128, 256), (7, 8, 128, 512)]
    if mode == "full":
        w4a4_shapes += [
            (2, 2, 64, 80),
            (1, 8, 1024, 2048),
            (7, 8, 2048, 512),
            (8, 1, 2048, 512),
            (56, 1, 2048, 512),
        ]
    for m, top_k, n, k in w4a4_shapes:
        count += _w4a4_constant_case(ops, m, top_k, n, k)
        # Keep random source tensors at bounded memory for the two exact target
        # shapes; packed constant tests still cover their full launch geometry.
        if n <= 128:
            count += _w4a4_random_case(ops, m, top_k, n, k)
    count += _cuda_graph_case(ops)
    count += _torch_compile_case(ops)
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    count = run(ops, args.mode)
    print(f"grouped-moe-gemv {args.backend} {args.mode}: passed {count}/{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
