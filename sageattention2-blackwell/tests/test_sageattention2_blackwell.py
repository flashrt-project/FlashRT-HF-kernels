#!/usr/bin/env python3
"""Correctness tests for sageattention2-blackwell."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import math
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "sageattention2-blackwell"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    @staticmethod
    def padded_k64(seqlen_k: int) -> int:
        return ((int(seqlen_k) + 63) // 64) * 64

    @staticmethod
    def q_scale_elems(batch: int, seqlen_q: int, q_heads: int) -> int:
        return int(batch) * int(q_heads) * ((int(seqlen_q) + 31) // 32)

    @staticmethod
    def k_scale_elems(batch: int, seqlen_k: int, kv_heads: int) -> int:
        return int(batch) * int(kv_heads) * ((int(seqlen_k) + 63) // 64)

    @classmethod
    def q_thread_scale_elems(cls, batch, seqlen_q, q_heads):
        return cls.q_scale_elems(batch, seqlen_q, q_heads) * 8

    @classmethod
    def k_thread_scale_elems(cls, batch, seqlen_k, kv_heads):
        return cls.k_scale_elems(batch, seqlen_k, kv_heads) * 4

    @staticmethod
    def v_scale_elems(batch: int, kv_heads: int) -> int:
        return int(batch) * int(kv_heads) * 128

    def quantize_q_bf16_d128(self, q, q_i8=None, q_scale=None):
        q_i8 = torch.empty_like(q, dtype=torch.int8) if q_i8 is None else q_i8
        if q_scale is None:
            q_scale = torch.empty((self.q_scale_elems(q.shape[0], q.shape[1], q.shape[2]),), device=q.device, dtype=torch.float32)
        self._ops.quantize_q_bf16_d128(q, q_i8, q_scale)
        return q_i8, q_scale

    def quantize_k_bf16_d128(self, k, k_i8=None, k_scale=None):
        k_i8 = torch.empty_like(k, dtype=torch.int8) if k_i8 is None else k_i8
        if k_scale is None:
            k_scale = torch.empty((self.k_scale_elems(k.shape[0], k.shape[1], k.shape[2]),), device=k.device, dtype=torch.float32)
        self._ops.quantize_k_bf16_d128(k, k_i8, k_scale)
        return k_i8, k_scale

    def quantize_qk_bf16_d128(self, q, k, q_i8=None, k_i8=None, q_scale=None, k_scale=None, *, qk_quant_granularity="per_warp"):
        per_thread = qk_quant_granularity == "per_thread"
        q_i8 = torch.empty_like(q, dtype=torch.int8) if q_i8 is None else q_i8
        k_i8 = torch.empty_like(k, dtype=torch.int8) if k_i8 is None else k_i8
        qn = self.q_thread_scale_elems(*q.shape[:3]) if per_thread else self.q_scale_elems(*q.shape[:3])
        kn = self.k_thread_scale_elems(*k.shape[:3]) if per_thread else self.k_scale_elems(*k.shape[:3])
        q_scale = torch.empty((qn,), device=q.device, dtype=torch.float32) if q_scale is None else q_scale
        k_scale = torch.empty((kn,), device=k.device, dtype=torch.float32) if k_scale is None else k_scale
        if per_thread:
            self._ops.quantize_qk_per_thread_bf16_d128(q, k, q_i8, k_i8, q_scale, k_scale)
        else:
            self._ops.quantize_q_bf16_d128(q, q_i8, q_scale)
            self._ops.quantize_k_bf16_d128(k, k_i8, k_scale)
        return q_i8, k_i8, q_scale, k_scale

    def quantize_v_fp16_bf16_d128(self, v, v_half=None):
        v_half = torch.empty_like(v, dtype=torch.float16) if v_half is None else v_half
        self._ops.quantize_v_fp16_bf16_d128(v, v_half)
        return v_half

    def quantize_v_fp8_bf16_d128(self, v, v_fp8_tpp=None, v_scale=None, v_tpp_bf16=None):
        if v_fp8_tpp is None:
            v_fp8_tpp = torch.empty((v.shape[0], 128, v.shape[2], self.padded_k64(v.shape[1])), device=v.device, dtype=torch.int8)
        if v_scale is None:
            v_scale = torch.empty((self.v_scale_elems(v.shape[0], v.shape[2]),), device=v.device, dtype=torch.float32)
        if v_tpp_bf16 is None:
            v_tpp_bf16 = torch.empty_like(v_fp8_tpp, dtype=torch.bfloat16)
        self._ops.quantize_v_fp8_native_bf16_d128(v, v_tpp_bf16, v_fp8_tpp, v_scale)
        return v_fp8_tpp, v_scale

    def allocate_workspace(self, q, k, v, *, fp8v=True, qk_quant_granularity="per_warp"):
        per_thread = qk_quant_granularity == "per_thread"
        qn = self.q_thread_scale_elems(*q.shape[:3]) if per_thread else self.q_scale_elems(*q.shape[:3])
        kn = self.k_thread_scale_elems(*k.shape[:3]) if per_thread else self.k_scale_elems(*k.shape[:3])
        common = dict(
            q_i8=torch.empty_like(q, dtype=torch.int8),
            k_i8=torch.empty_like(k, dtype=torch.int8),
            q_scale=torch.empty((qn,), device=q.device, dtype=torch.float32),
            k_scale=torch.empty((kn,), device=k.device, dtype=torch.float32),
            out=torch.empty_like(q),
            v_half=None,
            v_tpp_bf16=None,
            v_fp8_tpp=None,
            v_scale=None,
            qk_quant_granularity=qk_quant_granularity,
        )
        if fp8v:
            common["v_tpp_bf16"] = torch.empty((v.shape[0], 128, v.shape[2], self.padded_k64(v.shape[1])), device=v.device, dtype=torch.bfloat16)
            common["v_fp8_tpp"] = torch.empty((v.shape[0], 128, v.shape[2], self.padded_k64(v.shape[1])), device=v.device, dtype=torch.int8)
            common["v_scale"] = torch.empty((self.v_scale_elems(v.shape[0], v.shape[2]),), device=v.device, dtype=torch.float32)
        else:
            common["v_half"] = torch.empty_like(v, dtype=torch.float16)
        return SimpleNamespace(**common)

    def sage2_qk_int8_sv_f16_bf16_d128(self, q_i8, k_i8, v_half, q_scale, k_scale, *, softmax_scale=None, causal=False, out=None, qk_quant_granularity="per_warp"):
        out = torch.empty_like(q_i8, dtype=torch.bfloat16) if out is None else out
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(128)
        op = self._ops.sage2_qk_int8_pt_sv_f16_bf16_d128 if qk_quant_granularity == "per_thread" else self._ops.sage2_qk_int8_sv_f16_bf16_d128
        op(q_i8, k_i8, v_half, q_scale, k_scale, float(softmax_scale), bool(causal), out)
        return out

    def sage2_qk_int8_sv_f8_bf16_d128(self, q_i8, k_i8, v_fp8_tpp, q_scale, k_scale, v_scale, *, softmax_scale=None, causal=False, out=None, qk_quant_granularity="per_warp"):
        out = torch.empty_like(q_i8, dtype=torch.bfloat16) if out is None else out
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(128)
        op = self._ops.sage2_qk_int8_pt_sv_f8_bf16_d128 if qk_quant_granularity == "per_thread" else self._ops.sage2_qk_int8_sv_f8_bf16_d128
        op(q_i8, k_i8, v_fp8_tpp, q_scale, k_scale, v_scale, float(softmax_scale), bool(causal), out)
        return out

    def sage2_prefill_f16_bf16_d128(self, q, k, v, *, softmax_scale=None, causal=False, out=None, workspace=None, qk_quant_granularity="per_warp"):
        workspace = self.allocate_workspace(q, k, v, fp8v=False, qk_quant_granularity=qk_quant_granularity) if workspace is None else workspace
        if qk_quant_granularity == "per_thread":
            q_i8, k_i8, q_scale, k_scale = self.quantize_qk_bf16_d128(q, k, workspace.q_i8, workspace.k_i8, workspace.q_scale, workspace.k_scale, qk_quant_granularity=qk_quant_granularity)
        else:
            q_i8, q_scale = self.quantize_q_bf16_d128(q, workspace.q_i8, workspace.q_scale)
            k_i8, k_scale = self.quantize_k_bf16_d128(k, workspace.k_i8, workspace.k_scale)
        v_half = self.quantize_v_fp16_bf16_d128(v, workspace.v_half)
        out = workspace.out if out is None else out
        return self.sage2_qk_int8_sv_f16_bf16_d128(q_i8, k_i8, v_half, q_scale, k_scale, softmax_scale=softmax_scale, causal=causal, out=out, qk_quant_granularity=qk_quant_granularity)

    def sage2_prefill_fp8v_bf16_d128(self, q, k, v, *, softmax_scale=None, causal=False, out=None, workspace=None, qk_quant_granularity="per_warp"):
        workspace = self.allocate_workspace(q, k, v, fp8v=True, qk_quant_granularity=qk_quant_granularity) if workspace is None else workspace
        if qk_quant_granularity == "per_thread":
            q_i8, k_i8, q_scale, k_scale = self.quantize_qk_bf16_d128(q, k, workspace.q_i8, workspace.k_i8, workspace.q_scale, workspace.k_scale, qk_quant_granularity=qk_quant_granularity)
        else:
            q_i8, q_scale = self.quantize_q_bf16_d128(q, workspace.q_i8, workspace.q_scale)
            k_i8, k_scale = self.quantize_k_bf16_d128(k, workspace.k_i8, workspace.k_scale)
        v_fp8_tpp, v_scale = self.quantize_v_fp8_bf16_d128(v, workspace.v_fp8_tpp, workspace.v_scale, workspace.v_tpp_bf16)
        out = workspace.out if out is None else out
        return self.sage2_qk_int8_sv_f8_bf16_d128(q_i8, k_i8, v_fp8_tpp, q_scale, k_scale, v_scale, softmax_scale=softmax_scale, causal=causal, out=out, qk_quant_granularity=qk_quant_granularity)


def _preload_cublaslt() -> None:
    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    library = ctypes.util.find_library("cublasLt")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    suffix = "a" if major >= 12 else ""
    return f"{major}.{minor}{suffix}"


def load_source_ops():
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "sageattention2_blackwell_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "sage2_blackwell.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-DCUDA_KERNEL",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("sageattention2_blackwell")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_inputs(batch: int, seqlen: int, q_heads: int, kv_heads: int):
    q = (torch.randn(batch, seqlen, q_heads, 128, device="cuda", dtype=torch.float32) * 0.35).to(torch.bfloat16)
    k = (torch.randn(batch, seqlen, kv_heads, 128, device="cuda", dtype=torch.float32) * 0.35).to(torch.bfloat16)
    v = (torch.randn(batch, seqlen, kv_heads, 128, device="cuda", dtype=torch.float32) * 0.35).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v.contiguous()


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    q_t = q.transpose(1, 2).float()
    if q.shape[2] != k.shape[2]:
        repeat = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2).to(torch.bfloat16).contiguous()


def stats(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = (got.float() - ref.float()).abs().flatten()
    got_f = got.float().flatten()
    ref_f = ref.float().flatten()
    p99_src = diff
    if p99_src.numel() > 8_000_000:
        stride = (p99_src.numel() + 8_000_000 - 1) // 8_000_000
        p99_src = p99_src[::stride]
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(torch.quantile(p99_src, 0.99).item()),
        "cos": float(F.cosine_similarity(got_f, ref_f, dim=0).item()),
    }


def run_case(ops, name: str, batch: int, seqlen: int, q_heads: int, kv_heads: int, causal: bool, use_fp8v: bool, granularity: str = "per_warp"):
    q, k, v = make_inputs(batch, seqlen, q_heads, kv_heads)
    ref = reference(q, k, v, causal)
    if use_fp8v:
        got = ops.sage2_prefill_fp8v_bf16_d128(q, k, v, causal=causal, qk_quant_granularity=granularity)
    else:
        got = ops.sage2_prefill_f16_bf16_d128(q, k, v, causal=causal, qk_quant_granularity=granularity)
    torch.cuda.synchronize()
    s = stats(got, ref)
    min_cos = 0.9985 if seqlen <= 512 else 0.998
    if s["cos"] < min_cos or s["p99_abs"] > 0.25:
        raise AssertionError(f"{name} failed: {s}")
    print(
        f"PASS {name}: max_abs={s['max_abs']:.6f} mean_abs={s['mean_abs']:.6f} "
        f"p99_abs={s['p99_abs']:.6f} cos={s['cos']:.8f}"
    )


def _sage_round_i8(x: torch.Tensor) -> torch.Tensor:
    rounded = torch.where(x >= 0, x + 0.5, x - 0.5).trunc()
    return rounded.clamp(-127, 127).to(torch.int8)


def per_thread_reference(x: torch.Tensor, is_q: bool) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seqlen, heads, dim = x.shape
    tile_tokens, groups = (32, 8) if is_q else (64, 4)
    out = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((bsz, heads, math.ceil(seqlen / tile_tokens), groups), device=x.device, dtype=torch.float32)
    xf = x.float()
    for b in range(bsz):
        for h in range(heads):
            for tile in range(math.ceil(seqlen / tile_tokens)):
                base = tile * tile_tokens
                for group in range(groups):
                    if is_q:
                        positions = [base + group + 8 * i for i in range(4)]
                    else:
                        positions = [base + 2 * group + pair + 8 * i for i in range(8) for pair in range(2)]
                    positions = [p for p in positions if p < seqlen]
                    if not positions:
                        scales[b, h, tile, group] = 1.0e-7 / 127.0 + 1.0e-7
                        continue
                    vals = xf[b, positions, h, :]
                    scale = vals.abs().amax().clamp_min(1.0e-7) / 127.0 + 1.0e-7
                    scales[b, h, tile, group] = scale
                    out[b, positions, h, :] = _sage_round_i8(vals / scale)
    return out, scales.flatten()


def run_quantization_contract_gate(ops) -> None:
    q, k, _ = make_inputs(1, 70, 8, 4)
    old_q, old_qs = ops.quantize_q_bf16_d128(q)
    old_k, old_ks = ops.quantize_k_bf16_d128(k)
    new_q, new_k, new_qs, new_ks = ops.quantize_qk_bf16_d128(q, k)
    torch.cuda.synchronize()
    if not all(torch.equal(a, b) for a, b in ((old_q, new_q), (old_k, new_k), (old_qs, new_qs), (old_ks, new_ks))):
        raise AssertionError("fused per-warp Q/K producer changed the existing numerical contract")

    pt_q, pt_k, pt_qs, pt_ks = ops.quantize_qk_bf16_d128(q, k, qk_quant_granularity="per_thread")
    ref_q, ref_qs = per_thread_reference(q, True)
    ref_k, ref_ks = per_thread_reference(k, False)
    torch.cuda.synchronize()
    if not torch.equal(pt_q, ref_q) or not torch.equal(pt_k, ref_k):
        raise AssertionError("per-thread INT8 producer differs from the SageAttention grouping/rounding contract")
    if not torch.allclose(pt_qs, ref_qs, rtol=1e-6, atol=1e-8) or not torch.allclose(pt_ks, ref_ks, rtol=1e-6, atol=1e-8):
        raise AssertionError("per-thread scales differ from the SageAttention contract")
    print("PASS quantization_contract: combined per-warp bitwise; per-thread official grouping/rounding")


def run_v_producer_contract_gate(ops) -> None:
    _, _, v = make_inputs(1, 70, 4, 4)
    workspace = ops.allocate_workspace(v, v, v, fp8v=True)
    ops.quantize_v_fp8_bf16_d128(
        v, workspace.v_fp8_tpp, workspace.v_scale, workspace.v_tpp_bf16
    )
    padded = ops.padded_k64(v.shape[1])
    out_pos = torch.arange(padded, device=v.device)
    lane = out_pos % 16
    inverse_lane = torch.where(
        lane < 2, lane,
        torch.where(
            lane < 4, lane + 6,
            torch.where(
                lane < 6, lane - 2,
                torch.where(
                    lane < 8, lane + 4,
                    torch.where(
                        lane < 10, lane - 4,
                        torch.where(lane < 12, lane + 2, torch.where(lane < 14, lane - 6, lane)),
                    ),
                ),
            ),
        ),
    )
    source_pos = (out_pos // 16) * 16 + inverse_lane
    expected = torch.zeros_like(workspace.v_tpp_bf16)
    valid = source_pos < v.shape[1]
    expected[..., valid] = v[:, source_pos[valid], :, :].permute(0, 3, 2, 1)
    if not torch.equal(workspace.v_tpp_bf16, expected):
        raise AssertionError("V transpose/pad/permutation differs from the native contract")
    expected_scale = (
        expected.float().abs().amax(dim=-1).clamp_min(1.0e-7) / 448.0
    ).permute(0, 2, 1).flatten()
    if not torch.equal(workspace.v_scale, expected_scale):
        raise AssertionError("V per-channel FP8 scales differ from the native contract")
    decoded = workspace.v_fp8_tpp.view(torch.float8_e4m3fn).float()
    reconstructed = decoded * workspace.v_scale.reshape(1, v.shape[2], 128).permute(0, 2, 1).unsqueeze(-1)
    max_error = float((reconstructed - expected.float()).abs().max().item())
    max_allowed = float(expected_scale.max().item()) * 16.0 + 1.0e-6
    if max_error > max_allowed:
        raise AssertionError(
            f"V FP8 producer reconstruction error is too large: {max_error} > {max_allowed}"
        )
    print(f"PASS v_producer_contract: native layout/scales exact, max reconstruction error={max_error:.6f}")


def run_static_workspace_gate(ops, granularity: str = "per_warp") -> None:
    q, k, v = make_inputs(1, 128, 24, 24)
    workspace = ops.allocate_workspace(q, k, v, fp8v=True, qk_quant_granularity=granularity)
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            workspace.q_i8, workspace.k_i8, workspace.q_scale,
            workspace.k_scale, workspace.v_tpp_bf16, workspace.v_fp8_tpp, workspace.v_scale,
            workspace.out,
        )
    )
    eager = ops.sage2_prefill_fp8v_bf16_d128(q, k, v, workspace=workspace, qk_quant_granularity=granularity).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ops.sage2_prefill_fp8v_bf16_d128(q, k, v, workspace=workspace, qk_quant_granularity=granularity)
    graph.replay()
    torch.cuda.synchronize()
    if pointers != tuple(
        tensor.data_ptr()
        for tensor in (
            workspace.q_i8, workspace.k_i8, workspace.q_scale,
            workspace.k_scale, workspace.v_tpp_bf16, workspace.v_fp8_tpp, workspace.v_scale,
            workspace.out,
        )
    ):
        raise AssertionError("workspace pointers changed across graph replay")
    if captured.data_ptr() != workspace.out.data_ptr() or not torch.equal(eager, captured):
        raise AssertionError("workspace CUDA Graph replay is not deterministic")
    print(f"PASS static_workspace_cuda_graph[{granularity}]: stable pointers and bitwise replay")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, _minor = torch.cuda.get_device_capability(0)
    if major < 12:
        raise SystemExit("sageattention2-blackwell requires Blackwell-class CUDA capability")

    torch.manual_seed(2026)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    run_quantization_contract_gate(ops)
    run_v_producer_contract_gate(ops)
    cases = [
        ("wan_noncausal_s128_f16v", 1, 128, 24, 24, False, False),
        ("qwen_causal_gqa_s128_f16v", 1, 128, 32, 8, True, False),
    ]
    if args.mode == "full":
        cases.extend(
            [
                ("wan_noncausal_s256_fp8v", 1, 256, 24, 24, False, True),
                ("qwen_causal_gqa_s256_fp8v", 1, 256, 32, 8, True, True),
                ("qwen_causal_gqa_s512_f16v", 1, 512, 32, 8, True, False),
                ("wan_noncausal_s3600_partial_f16v", 1, 3600, 24, 24, False, False),
                ("wan_noncausal_s3600_partial_fp8v", 1, 3600, 24, 24, False, True),
                ("wan_noncausal_s5070_partial_f16v", 1, 5070, 24, 24, False, False),
                ("qwen_causal_gqa_s3600_partial_f16v", 1, 3600, 32, 8, True, False),
                ("wan_noncausal_s5070_partial_f16v_pt", 1, 5070, 24, 24, False, False, "per_thread"),
                ("wan_noncausal_s3600_partial_fp8v_pt", 1, 3600, 24, 24, False, True, "per_thread"),
                ("qwen_causal_gqa_s512_f16v_pt", 1, 512, 32, 8, True, False, "per_thread"),
            ]
        )
    for case in cases:
        run_case(ops, *case)
    run_static_workspace_gate(ops)
    run_static_workspace_gate(ops, "per_thread")


if __name__ == "__main__":
    main()
