#!/usr/bin/env python3
"""Correctness and CUDA Graph gates for sageattention3-blackwell."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "sageattention3-blackwell"
REGISTRATION_INCLUDE = (
    ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject"
    / "templates" / "torch"
)
UPSTREAM = ROOT.parent / "ltx25_dev" / "SageAttention" / "sageattention3_blackwell"


class SourceOps:
    def __init__(self, namespace: str):
        self.ops = getattr(torch.ops, namespace)

    def quantize_q_fp4_nhd(self, x, packed, sf):
        self.ops.quantize_q_fp4_nhd(x, packed, sf)

    def quantize_k_fp4_nhd(self, x, packed, sf):
        self.ops.quantize_k_fp4_nhd(x, packed, sf)

    def quantize_v_fp4_nhd(self, x, packed, sf):
        self.ops.quantize_v_fp4_nhd(x, packed, sf)

    def fused(self, q, k, v, ws):
        base, kc, qm, km, db = ws
        b, length, h, d = q.shape; groups = qm.shape[2]
        torch.mean(k, dim=1, out=km)
        self.ops.quantize_q_fp4_centered_nhd(q, qm, base[0], base[3])
        self.ops.quantize_k_fp4_centered_nhd(k, km, base[1], base[4], kc)
        self.ops.quantize_v_fp4_nhd(v, base[2], base[5])
        b, h, g, d = qm.shape
        lp = kc.shape[2]
        torch.bmm(qm.view(b*h,g,d), kc.view(b*h,lp,d).transpose(1,2), out=db.view(b*h,g,lp))
        return self.attention(base, db, q.shape[1], True)[:, :q.shape[1]]

    def attention(self, ws, delta_s, unpadded_k, per_block_mean):
        self.ops.blockscaled_fp4_attention_static(
            ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], delta_s,
            int(unpadded_k), 1.0 / math.sqrt(ws[9].shape[-1]), False,
            bool(per_block_mean), ws[9].dtype == torch.bfloat16,
            ws[9], ws[10], ws[11],
        )
        return ws[9].transpose(1, 2)


class InstalledOps(SourceOps):
    def __init__(self, module):
        self.module = module
        self.ops = module.ops


def load_installed_module(artifact: str | None):
    if not artifact:
        return importlib.import_module("sageattention3_blackwell")

    artifact_path = Path(artifact)
    flat_init = artifact_path / "__init__.py"
    if flat_init.is_file():
        spec = importlib.util.spec_from_file_location(
            "sageattention3_blackwell",
            flat_init,
            submodule_search_locations=[str(artifact_path)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load artifact entry: {flat_init}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    sys.path.insert(0, str(artifact_path))
    try:
        return importlib.import_module("sageattention3_blackwell")
    finally:
        sys.path.remove(str(artifact_path))


def load_source_ops():
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    namespace = "sageattention3_blackwell_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "blackwell" / "api.cu"),
            str(PACKAGE / "csrc" / "quantization" / "fp4_quantization_4d.cu"),
        ],
        extra_include_paths=[
            str(PACKAGE / "csrc"),
            str(PACKAGE / "csrc" / "blackwell"),
            str(PACKAGE / "csrc" / "quantization"),
            str(REGISTRATION_INCLUDE),
            str(UPSTREAM / "csrc" / "cutlass" / "include"),
        ],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "--use_fast_math", "-DNDEBUG", "-DQBLKSIZE=128",
            "-DKBLKSIZE=128", "-DCTA256", "-DDQINRMEM", "-DCUDA_KERNEL",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        ],
        is_python_module=False,
        verbose=False,
    )
    return SourceOps(namespace)


def preprocess(q, k, v, per_block_mean):
    # Inputs and outputs are NHD; pad after K centering to preserve the contract.
    b, s, h, d = q.shape
    padded = (s + 127) // 128 * 128
    qh = q.transpose(1, 2).contiguous()
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()
    kh = kh - kh.mean(dim=-2, keepdim=True)
    if padded != s:
        qh = torch.nn.functional.pad(qh, (0, 0, 0, padded - s))
        kh = torch.nn.functional.pad(kh, (0, 0, 0, padded - s))
        vh = torch.nn.functional.pad(vh, (0, 0, 0, padded - s))
    if per_block_mean:
        qm = qh.reshape(b, h, padded // 128, 128, d).mean(dim=-2)
        qh = qh - qm.repeat_interleave(128, dim=-2)
    else:
        qm = qh.mean(dim=-2, keepdim=True)
        qh = qh - qm
    delta_s = torch.matmul(qm, kh.transpose(-2, -1)).float().contiguous()
    return (
        qh.transpose(1, 2).contiguous(),
        kh.transpose(1, 2).contiguous(),
        vh.transpose(1, 2).contiguous(),
        delta_s,
        qh,
        kh,
        vh,
    )


def alloc(q):
    b, s, h, d = q.shape
    out_nhd = torch.empty((b, s, h, d), device=q.device, dtype=q.dtype)
    return (
        torch.empty((b, h, s, d // 2), device=q.device, dtype=torch.uint8),
        torch.empty((b, h, s, d // 2), device=q.device, dtype=torch.uint8),
        torch.empty((b, h, d, s // 2), device=q.device, dtype=torch.uint8),
        torch.empty((b, h, s, d // 16), device=q.device, dtype=torch.float8_e4m3fn),
        torch.empty((b, h, s, d // 16), device=q.device, dtype=torch.float8_e4m3fn),
        torch.empty((b, h, d, s // 16), device=q.device, dtype=torch.float8_e4m3fn),
        q, q, q,  # retain source tensors for a compact tuple contract
        out_nhd.transpose(1, 2),
        torch.empty((b, h, s), device=q.device, dtype=torch.float32),
        torch.empty((1,), device=q.device, dtype=torch.int32),
    )


def alloc_fused(q):
    b, length, h, d = q.shape
    lp = (length + 127) // 128 * 128
    padded = torch.empty((b, lp, h, d), device=q.device, dtype=q.dtype)
    base = list(alloc(padded))
    groups = lp // 128
    return (
        base,
        torch.empty((b,h,lp,d),device=q.device,dtype=q.dtype),
        torch.empty((b,h,groups,d),device=q.device,dtype=q.dtype),
        torch.empty((b,h,d),device=q.device,dtype=q.dtype),
        torch.empty((b,h,groups,lp),device=q.device,dtype=q.dtype),
    )


def run_fused_prep_case(ops, s, d):
    q=torch.randn((1,s,2,d),device="cuda",dtype=torch.bfloat16)
    k=torch.randn_like(q); v=torch.randn_like(q)
    ws=alloc_fused(q); got=ops.fused(q,k,v,ws)
    qn,kn,vn,ds,*_=preprocess(q,k,v,True); legacy=list(alloc(qn))
    ops.quantize_q_fp4_nhd(qn,legacy[0],legacy[3]); ops.quantize_k_fp4_nhd(kn,legacy[1],legacy[4]); ops.quantize_v_fp4_nhd(vn,legacy[2],legacy[5])
    ref=ops.attention(legacy,ds,s,True)[:,:s]
    cos=cosine(got,ref); max_abs=(got.float()-ref.float()).abs().max().item()
    if cos < 0.99999: raise AssertionError(f"fused prep parity failed: cos={cos} max={max_abs}")
    graph=torch.cuda.CUDAGraph(); torch.cuda.synchronize()
    with torch.cuda.graph(graph): captured=ops.fused(q,k,v,ws)
    graph.replay(); first=captured.clone(); graph.replay()
    if not torch.equal(first,captured): raise AssertionError("fused prep graph replay is not bitwise")
    print(f"PASS fused_prep S={s} D={d}: cos={cos:.8f} max={max_abs:.6f}, graph=bitwise")


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item())


def run_case(ops, s, d, per_block_mean):
    dtype = torch.bfloat16
    q = torch.randn((1, s, 2, d), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    qn, kn, vn, delta_s, qh, kh, vh = preprocess(q, k, v, per_block_mean)
    ws = list(alloc(qn))
    ops.quantize_q_fp4_nhd(qn, ws[0], ws[3])
    ops.quantize_k_fp4_nhd(kn, ws[1], ws[4])
    ops.quantize_v_fp4_nhd(vn, ws[2], ws[5])
    got = ops.attention(ws, delta_s, s, per_block_mean)
    ref = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh).transpose(1, 2)
    cos = cosine(got, ref)
    if cos < 0.97:
        raise AssertionError(f"Sage3 cosine too low: {cos:.8f}")

    pointers = tuple(t.data_ptr() for t in ws[:6] + ws[9:])
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        ops.quantize_q_fp4_nhd(qn, ws[0], ws[3])
        ops.quantize_k_fp4_nhd(kn, ws[1], ws[4])
        ops.quantize_v_fp4_nhd(vn, ws[2], ws[5])
        captured = ops.attention(ws, delta_s, s, per_block_mean)
    graph.replay()
    first = captured.clone()
    graph.replay()
    if not torch.equal(first, captured):
        raise AssertionError("CUDA Graph replay is not bitwise deterministic")
    if pointers != tuple(t.data_ptr() for t in ws[:6] + ws[9:]):
        raise AssertionError("workspace pointers changed")
    print(f"PASS S={s} D={d} block_mean={per_block_mean}: cos={cos:.8f}, graph=bitwise")


def run_invalid_contract_gate(ops, d=128):
    q = torch.randn((1, 128, 2, d), device="cuda", dtype=torch.bfloat16)
    qn, kn, vn, _delta_s, *_ = preprocess(q, q, q, False)
    ws = list(alloc(qn))
    ops.quantize_q_fp4_nhd(qn, ws[0], ws[3])
    ops.quantize_k_fp4_nhd(kn, ws[1], ws[4])
    ops.quantize_v_fp4_nhd(vn, ws[2], ws[5])
    wrong_delta = torch.empty((1, 2, 2, 128), device="cuda", dtype=torch.float32)
    try:
        ops.attention(ws, wrong_delta, 128, False)
    except RuntimeError:
        print("PASS invalid_delta_s: explicit RuntimeError")
    else:
        raise AssertionError("invalid delta_s did not raise")

    wrong_dtype = torch.empty((1, 2, 1, 128), device="cuda", dtype=torch.float16)
    try:
        ops.attention(ws, wrong_dtype, 128, False)
    except RuntimeError:
        print("PASS invalid_delta_dtype: explicit RuntimeError")
    else:
        raise AssertionError("invalid delta_s dtype did not raise")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    if torch.cuda.get_device_capability(0)[0] != 12:
        raise SystemExit("sageattention3-blackwell requires SM120/SM121")
    if args.backend == "source":
        ops = load_source_ops()
    else:
        module = load_installed_module(args.artifact)
        expected = {
            "layouts": ("NHD",),
            "caller_owned_workspace": True,
            "cuda_graph_safe": True,
            "fused_prep": True,
            "delta_dtypes": ("float32", "bfloat16"),
            "d128_min_cuda": "13.0",
        }
        actual = module.capabilities()
        for key, value in expected.items():
            if actual.get(key) != value:
                raise AssertionError(f"capability {key}: {actual.get(key)!r} != {value!r}")
        supported_head_dims = tuple(actual["head_dims"])
        ops = InstalledOps(module)
    if args.backend == "source":
        supported_head_dims = (64, 128)
    cases = [(128, d) for d in supported_head_dims]
    if args.mode == "full":
        cases += [(2688, 64)]
        if 128 in supported_head_dims:
            cases += [(6144, 128), (24576, 128)]
    for s, d in cases:
        for per_block_mean in (False, True):
            run_case(ops, s, d, per_block_mean)
    run_invalid_contract_gate(ops, supported_head_dims[-1])
    run_fused_prep_case(ops, 256, supported_head_dims[-1])
    if args.mode == "full":
        run_fused_prep_case(ops, 384, 64)
        if 128 in supported_head_dims:
            run_fused_prep_case(ops, 5070, 128)
    if args.backend == "installed" and 128 not in supported_head_dims:
        q = torch.randn((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
        try:
            module.allocate_fused_workspace(q, q, q)
        except RuntimeError as exc:
            if "CUDA" not in str(exc) or "64" not in str(exc):
                raise AssertionError(f"unexpected D128 rejection: {exc}") from exc
            print("PASS D128: CUDA 12.8 artifact rejects unsupported performance tier")
        else:
            raise AssertionError("CUDA 12.8 artifact did not reject D128")


if __name__ == "__main__":
    main()
