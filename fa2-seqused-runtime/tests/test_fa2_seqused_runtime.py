#!/usr/bin/env python3
"""Correctness tests for fa2-seqused-runtime (static-buffer FA2 forward)."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "fa2-seqused-runtime"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)
CUTLASS_INCLUDE = Path(os.environ.get(
    "FA2_CUTLASS_INCLUDE", "/data/third_party/cutlass-3.6.0/include"
))

SUPPORTED_HEAD_DIMS = tuple(range(8, 257, 8))
SPLIT_HEAD_DIMS = tuple(range(40, 129, 8)) + tuple(range(232, 257, 8))

_FA2_SOURCES = [
    "torch-ext/torch_binding.cpp",
    "csrc/fa2_wrapper.cu",
    "csrc/fa2_wrapper_causal.cu",
    "csrc/flash_attn/flash_fwd_hdim64_fp16_sm80.cu",
    "csrc/flash_attn/flash_fwd_hdim64_bf16_sm80.cu",
    "csrc/flash_attn/flash_fwd_hdim96_fp16_sm80.cu",
    "csrc/flash_attn/flash_fwd_hdim96_bf16_sm80.cu",
    "csrc/flash_attn/flash_fwd_hdim128_fp16_sm80.cu",
    "csrc/flash_attn/flash_fwd_hdim128_bf16_sm80.cu",
    "csrc/flash_attn/flash_fwd_hdim256_fp16_sm80.cu",
    "csrc/flash_attn/flash_fwd_hdim256_bf16_sm80.cu",
    "csrc/flash_attn/flash_fwd_split_hdim64_fp16_sm80.cu",
    "csrc/flash_attn/flash_fwd_split_hdim64_bf16_sm80.cu",
    "csrc/flash_attn/flash_fwd_split_hdim96_fp16_sm80.cu",
    "csrc/flash_attn/flash_fwd_split_hdim96_bf16_sm80.cu",
    "csrc/flash_attn/flash_fwd_split_hdim128_fp16_sm80.cu",
    "csrc/flash_attn/flash_fwd_split_hdim128_bf16_sm80.cu",
    "csrc/flash_attn/flash_fwd_split_hdim256_fp16_sm80.cu",
    "csrc/flash_attn/flash_fwd_split_hdim256_bf16_sm80.cu",
    "csrc/flash_attn/flash_fwd_hdim128_bf16_sm80_causal.cu",
    "csrc/flash_attn/flash_fwd_split_hdim128_bf16_sm80_causal.cu",
    "csrc/flash_attn/flash_fwd_hdim256_bf16_sm80_causal.cu",
    "csrc/flash_attn/flash_fwd_split_hdim256_bf16_sm80_causal.cu",
]

_FA2_CUDA_FLAGS = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-DFA2_HAS_HDIM_64=1",
    "-DFA2_HAS_HDIM_96=1",
    "-DFA2_HAS_HDIM_128=1",
    "-DFA2_HAS_HDIM_256=1",
    "-DFA2_HAS_FP16=1",
    "-DFA2_HAS_BF16=1",
]


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 12:
        return "12.0a"
    if (major, minor) == (11, 0):
        return "11.0a"
    return f"{major}.{minor}"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def allocate_outputs(self, q):
        out = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
        lse = torch.empty((q.shape[0], q.shape[2], q.shape[1]), device=q.device, dtype=torch.float32)
        return out, lse

    def forward(self, q, k, v, *, softmax_scale=None, causal=False, use_split_kv=False):
        out, lse = self.allocate_outputs(q)
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        self._ops.forward_static(
            q, k, v, out, lse, None, None, float(softmax_scale), bool(causal), 0
        )
        return out

    def forward_static(self, q, k, v, *, out, softmax_lse, workspace=None, softmax_scale=None, causal=False):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        lse_accum = None
        out_accum = None
        num_sms = 0
        if workspace is not None:
            lse_accum, out_accum, num_sms = workspace
        self._ops.forward_static(
            q, k, v, out, softmax_lse, lse_accum, out_accum,
            float(softmax_scale), bool(causal), int(num_sms),
        )
        return out

    def forward_seqused_static(self, q, k, v, seqused_k, *, out, softmax_lse, workspace=None, softmax_scale=None):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        lse_accum = None
        out_accum = None
        num_sms = 0
        if workspace is not None:
            lse_accum, out_accum, num_sms = workspace
        if lse_accum is not None:
            lse_accum.fill_(-torch.inf)
        self._ops.forward_seqused_static(
            q, k, v, seqused_k, out, softmax_lse, lse_accum, out_accum,
            float(softmax_scale), int(num_sms),
        )
        return out

    def allocate_workspace(self, q, k, *, num_sms=None):
        # The installed package computes the heuristic in Python; the source
        # backend does not expose it. Always return None so the no-split path
        # is exercised in source mode.
        return None


class InstalledOps:
    def __init__(self, module) -> None:
        self._module = module

    def allocate_outputs(self, q):
        return self._module.allocate_outputs(q)

    def forward(self, q, k, v, *, softmax_scale=None, causal=False, use_split_kv=True):
        return self._module.forward(q, k, v, softmax_scale=softmax_scale, causal=causal, use_split_kv=use_split_kv)

    def forward_static(self, q, k, v, *, out, softmax_lse, workspace=None, softmax_scale=None, causal=False):
        return self._module.forward_static(q, k, v, out=out, softmax_lse=softmax_lse, workspace=workspace, softmax_scale=softmax_scale, causal=causal)

    def forward_seqused_static(self, q, k, v, seqused_k, *, out, softmax_lse, workspace=None, softmax_scale=None):
        return self._module.forward_seqused_static(q, k, v, seqused_k, out=out, softmax_lse=softmax_lse, workspace=workspace, softmax_scale=softmax_scale)

    def allocate_workspace(self, q, k, *, num_sms=None):
        return self._module.allocate_workspace(q, k, num_sms=num_sms)


def _preload_cublaslt() -> None:
    import ctypes
    import ctypes.util

    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    library = ctypes.util.find_library("cublasLt")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    if not CUTLASS_INCLUDE.is_dir():
        raise RuntimeError(f"missing CUTLASS include path: {CUTLASS_INCLUDE}")
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "fa2_seqused_runtime_test"
    load(
        name=namespace,
        sources=[str(PACKAGE / s) for s in _FA2_SOURCES],
        extra_include_paths=[
            str(PACKAGE / "csrc"),
            str(PACKAGE / "csrc" / "flash_attn"),
            str(CUTLASS_INCLUDE),
            str(REGISTRATION_INCLUDE),
        ],
        extra_cflags=["-O3", "-DCUDA_KERNEL", "-DFA2_HAS_HDIM_128=1", "-DFA2_HAS_HDIM_256=1", "-DFA2_HAS_BF16=1", "-DFA2_HAS_FP16=1"],
        extra_cuda_cflags=["-DCUDA_KERNEL", *_FA2_CUDA_FLAGS],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return InstalledOps(importlib.import_module("fa2_seqused_runtime"))
    finally:
        if artifact:
            sys.path.remove(artifact)


def _reference(q, k, v, causal=False):
    hq, hkv = q.shape[2], k.shape[2]
    if hq != hkv:
        repeat = hq // hkv
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
    attn_mask = None
    if causal:
        sq, sk = q.shape[1], k.shape[1]
        q_idx = torch.arange(sq, device=q.device).view(sq, 1)
        k_idx = torch.arange(sk, device=q.device).view(1, sk)
        attn_mask = k_idx <= q_idx + sk - sq
    return F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
        attn_mask=attn_mask, is_causal=False,
    ).permute(0, 2, 1, 3)


def _assert_close(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    if actual.dtype == torch.float16:
        assert diff.max().item() <= 8e-3, f"max_abs {diff.max().item()}"
        assert diff.mean().item() <= 4e-4, f"mean_abs {diff.mean().item()}"
        assert cosine.item() >= 0.9999, f"cosine {cosine.item()}"
    else:
        assert diff.max().item() <= 6.5e-2, f"max_abs {diff.max().item()}"
        assert diff.mean().item() <= 3e-3, f"mean_abs {diff.mean().item()}"
        assert cosine.item() >= 0.999, f"cosine {cosine.item()}"


def _check_noncausal(ops, dtype, head_dim, shape):
    batch, sq, sk, hq, hkv = shape
    q = torch.randn(batch, sq, hq, head_dim, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(batch, sk, hkv, head_dim, device="cuda", dtype=dtype) * 0.5
    v = torch.randn_like(k)
    actual = ops.forward(q, k, v, use_split_kv=False)
    _assert_close(actual, _reference(q, k, v))
    print(f"PASS noncausal dtype={dtype} head_dim={head_dim} shape={shape}")


def _check_causal(ops, head_dim, seqlen):
    q = torch.randn(1, seqlen, 8, head_dim, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(1, seqlen, 2, head_dim, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k)
    actual = ops.forward(q, k, v, causal=True, use_split_kv=False)
    _assert_close(actual, _reference(q, k, v, causal=True))
    print(f"PASS causal head_dim={head_dim} seqlen={seqlen}")


def _check_seqused(ops):
    q = torch.randn(2, 17, 8, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(2, 513, 2, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k)
    used = torch.tensor([127, 513], device="cuda", dtype=torch.int32)
    out, lse = ops.allocate_outputs(q)
    ops.forward_seqused_static(q, k, v, used, out=out, softmax_lse=lse)
    refs = []
    for batch, n_used in enumerate((127, 513)):
        refs.append(_reference(q[batch : batch + 1], k[batch : batch + 1, :n_used], v[batch : batch + 1, :n_used]))
    _assert_close(out, torch.cat(refs, dim=0))
    print("PASS device seqused per-batch")


def _check_padded_strides(ops):
    def padded(shape):
        storage = torch.randn(*shape[:-1], shape[-1] + 8, device="cuda", dtype=torch.bfloat16)
        return storage[..., : shape[-1]]

    q = padded((1, 49, 8, 128))
    k = padded((1, 257, 2, 128))
    v = padded((1, 257, 2, 128))
    out = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
    lse = torch.empty((1, 8, 49), device="cuda", dtype=torch.float32)
    ops.forward_static(q, k, v, out=out, softmax_lse=lse)
    _assert_close(out, _reference(q, k, v))
    print("PASS aligned padded strides")


def _check_rejections(ops):
    bad_dim = 44
    q = torch.randn(1, 4, 4, bad_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 4, 4, bad_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    lse = torch.empty((1, 4, 4), device="cuda", dtype=torch.float32)
    try:
        ops.forward_static(q, k, v, out=out, softmax_lse=lse)
    except RuntimeError as exc:
        if "head_dim" not in str(exc):
            raise
    else:
        raise AssertionError("unbuilt head_dim must be rejected")

    q16 = torch.randn(1, 4, 4, 128, device="cuda", dtype=torch.float16)
    k16 = torch.randn(1, 4, 4, 128, device="cuda", dtype=torch.float16)
    v16 = torch.randn_like(k16)
    out16, lse16 = ops.allocate_outputs(q16)
    try:
        ops.forward_static(q16, k16, v16, out=out16, softmax_lse=lse16, causal=True)
    except RuntimeError as exc:
        if "causal v1 supports bf16" not in str(exc):
            raise
    else:
        raise AssertionError("fp16 causal must be rejected")

    storage = torch.randn(1, 8, 4, 129, device="cuda", dtype=torch.bfloat16)
    qs = storage[..., :128]
    ks = storage[..., :128]
    vs = storage[..., :128]
    outs = torch.empty_strided(qs.shape, qs.stride(), device=qs.device, dtype=qs.dtype)
    lses = torch.empty((1, 4, 8), device="cuda", dtype=torch.float32)
    try:
        ops.forward_static(qs, ks, vs, out=outs, softmax_lse=lses)
    except RuntimeError as exc:
        if "16-byte alignment" not in str(exc):
            raise
    else:
        raise AssertionError("misaligned head stride must be rejected")
    print("PASS rejections (unbuilt head_dim, fp16 causal, misaligned stride)")


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    if args.mode == "full":
        for dtype in (torch.float16, torch.bfloat16):
            for head_dim in (48, 64, 128):
                for shape in [(1, 17, 63, 16, 4), (2, 64, 129, 12, 4)]:
                    _check_noncausal(ops, dtype, head_dim, shape)
        for head_dim in (128, 256):
            for seqlen in (1, 17, 257):
                _check_causal(ops, head_dim, seqlen)
    else:
        _check_noncausal(ops, torch.bfloat16, 128, (1, 17, 63, 16, 4))
        _check_noncausal(ops, torch.float16, 64, (1, 17, 63, 16, 4))
        _check_causal(ops, 128, 17)
    _check_seqused(ops)
    _check_padded_strides(ops)
    _check_rejections(ops)
    if args.backend == "installed":
        if args.mode == "full":
            for head_dim in SPLIT_HEAD_DIMS:
                if head_dim not in (48, 64, 96, 128):
                    continue
                q = torch.randn(1, 1, 8, head_dim, device="cuda", dtype=torch.bfloat16) * 0.5
                k = torch.randn(1, 4096, 2, head_dim, device="cuda", dtype=torch.bfloat16) * 0.5
                v = torch.randn_like(k)
                workspace = ops.allocate_workspace(q, k)
                out, lse = ops.allocate_outputs(q)
                ops.forward_static(q, k, v, out=out, softmax_lse=lse, workspace=workspace)
                _assert_close(out, _reference(q, k, v))
            print("PASS split-KV workspace (installed)")
    print(f"PASS fa2-seqused-runtime {args.backend} mode={args.mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"passed": 1, "total": 1, "backend": args.backend}) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
