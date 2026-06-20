#!/usr/bin/env python3
"""Correctness tests for gated-delta-attention."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "gated-delta-attention"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)

D = 128
SHAPES = {
    "recurrent_h4": ("recurrent", 1, 1, 4),
    "inout_h4": ("inout", 1, 1, 4),
    "f32state_h4": ("f32state", 1, 1, 4),
    "chunk_s4_h4": ("chunk", 1, 4, 4),
    "chunk_smem_s4_h4": ("chunk_smem", 1, 4, 4),
    "recurrent_h48": ("recurrent", 1, 1, 48),
    "split_s4": ("split", 1, 4, 48),
    "gating_s4": ("gating", 1, 4, 48),
    "chunk_from_conv_s4": ("chunk_from_conv", 1, 4, 48),
    "wy_pipeline_s4": ("wy_pipeline", 1, 4, 48),
    "wy_pipeline_s65": ("wy_pipeline", 1, 65, 48),
}
MODES = {
    "smoke": ["recurrent_h4"],
    "headline": ["recurrent_h48", "chunk_s4_h4", "wy_pipeline_s65"],
    "full": list(SHAPES.keys()),
}


@dataclass
class Row:
    name: str
    kind: str
    B: int
    S: int
    H: int
    max_abs: float
    mean_abs: float
    p99_abs: float
    cosine: float
    passed: bool


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def recurrent(self, q, k, v, g, beta, state, use_qk_l2norm=True):
        out = torch.empty_like(q)
        self._ops.gated_delta_recurrent_bf16(q, k, v, g, beta, state, out, use_qk_l2norm)
        return out

    def inout(self, q, k, v, g, beta, state, use_qk_l2norm=True):
        out = torch.empty_like(q)
        state_out = torch.empty_like(state)
        self._ops.gated_delta_recurrent_inout_bf16(q, k, v, g, beta, state, state_out, out, use_qk_l2norm)
        return out, state_out

    def f32state(self, q, k, v, g, beta, state, use_qk_l2norm=True):
        out = torch.empty_like(q)
        self._ops.gated_delta_recurrent_f32state_bf16io(q, k, v, g, beta, state, out, use_qk_l2norm)
        return out

    def chunk(self, q, k, v, g, beta, state, use_qk_l2norm=True, smem=False):
        out = torch.empty_like(q)
        if smem:
            self._ops.gated_delta_chunk_smem_bf16(q, k, v, g, beta, state, out, use_qk_l2norm)
        else:
            self._ops.gated_delta_chunk_bf16(q, k, v, g, beta, state, out, use_qk_l2norm)
        return out

    def split_broadcast(self, conv_out):
        q = torch.empty((conv_out.shape[0], 48, D), device=conv_out.device, dtype=conv_out.dtype)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        self._ops.lin_split_qkv_broadcast_bf16(conv_out, q, k, v)
        return q, k, v

    def split_gqa(self, conv_out):
        q = torch.empty((conv_out.shape[0], 16, D), device=conv_out.device, dtype=conv_out.dtype)
        k = torch.empty_like(q)
        v = torch.empty((conv_out.shape[0], 48, D), device=conv_out.device, dtype=conv_out.dtype)
        self._ops.lin_split_qkv_gqa_bf16(conv_out, q, k, v)
        return q, k, v

    def split_q_gate(self, q_proj):
        q_pre = torch.empty((q_proj.shape[0], 24, 256), device=q_proj.device, dtype=q_proj.dtype)
        gate = torch.empty((q_proj.shape[0], 24 * 256), device=q_proj.device, dtype=q_proj.dtype)
        self._ops.split_q_gate_bf16(q_proj, q_pre, gate)
        return q_pre, gate

    def gating(self, a, b, neg, dt):
        g = torch.empty_like(a)
        beta = torch.empty_like(a)
        self._ops.gdn_gating_bf16(a, b, neg, dt, g, beta)
        return g, beta

    def gating_strided(self, a, b, neg, dt, rows, a_stride, b_stride):
        g = torch.empty((rows, 48), device=a.device, dtype=a.dtype)
        beta = torch.empty_like(g)
        self._ops.gdn_gating_strided_bf16(a, b, neg, dt, g, beta, a_stride, b_stride)
        return g, beta

    def chunk_from_conv(self, conv_out, a, b, neg, dt, state, use_qk_l2norm=True):
        out = torch.empty((conv_out.shape[0], 48, D), device=conv_out.device, dtype=conv_out.dtype)
        self._ops.gdn_chunk_from_conv_smem_bf16(conv_out, a, b, neg, dt, state, out, use_qk_l2norm)
        return out

    def wy_pipeline(self, q16, k16, v48, g, beta, state):
        S = q16.shape[0]
        chunks = (S + 63) // 64
        q16_l2 = torch.empty_like(q16)
        k16_l2 = torch.empty_like(k16)
        q_pack = torch.empty((chunks, 48, 64, D), device=q16.device, dtype=q16.dtype)
        k_pack = torch.empty((chunks, 16, 64, D), device=q16.device, dtype=q16.dtype)
        g_cumsum = torch.empty_like(g)
        A = torch.empty((chunks, 48, 64, 64), device=q16.device, dtype=torch.float32)
        Ai = torch.empty_like(A)
        w = torch.empty_like(v48)
        u = torch.empty_like(v48)
        h0 = torch.empty((chunks, 48, D, D), device=q16.device, dtype=q16.dtype)
        v_new = torch.empty_like(v48)
        out = torch.empty_like(v48)
        self._ops.gdn_wy_norm_cumsum_pack_qk_bf16(q16, k16, g, q16_l2, k16_l2, q_pack, k_pack, g_cumsum)
        self._ops.gdn_wy_kkt_b64_bf16(k16_l2, beta, g_cumsum, A)
        self._ops.gdn_wy_solve_tril_b64_f32(A, Ai, S)
        self._ops.gdn_wy_recompute_wu_b64_bf16(k16_l2, v48, beta, g_cumsum, Ai, w, u)
        self._ops.gdn_wy_chunk_h_b64_bf16(k16_l2, u, w, g_cumsum, state, h0, v_new)
        self._ops.gdn_wy_output_o_b64_bf16(q16_l2, k16_l2, v_new, h0, g_cumsum, out)
        return out


class InstalledOps:
    def __init__(self, mod) -> None:
        self._mod = mod

    def recurrent(self, q, k, v, g, beta, state, use_qk_l2norm=True):
        return self._mod.gated_delta_recurrent_bf16(
            q, k, v, g, beta, state, use_qk_l2norm=use_qk_l2norm
        )

    def inout(self, q, k, v, g, beta, state, use_qk_l2norm=True):
        return self._mod.gated_delta_recurrent_inout_bf16(
            q, k, v, g, beta, state, use_qk_l2norm=use_qk_l2norm
        )

    def f32state(self, q, k, v, g, beta, state, use_qk_l2norm=True):
        return self._mod.gated_delta_recurrent_f32state_bf16io(
            q, k, v, g, beta, state, use_qk_l2norm=use_qk_l2norm
        )

    def chunk(self, q, k, v, g, beta, state, use_qk_l2norm=True, smem=False):
        if smem:
            return self._mod.gated_delta_chunk_smem_bf16(
                q, k, v, g, beta, state, use_qk_l2norm=use_qk_l2norm
            )
        return self._mod.gated_delta_chunk_bf16(
            q, k, v, g, beta, state, use_qk_l2norm=use_qk_l2norm
        )

    def split_broadcast(self, conv_out):
        return self._mod.lin_split_qkv_broadcast_bf16(conv_out)

    def split_gqa(self, conv_out):
        return self._mod.lin_split_qkv_gqa_bf16(conv_out)

    def split_q_gate(self, q_proj):
        return self._mod.split_q_gate_bf16(q_proj)

    def gating(self, a, b, neg, dt):
        return self._mod.gdn_gating_bf16(a, b, neg, dt)

    def gating_strided(self, a, b, neg, dt, rows, a_stride, b_stride):
        return self._mod.gdn_gating_strided_bf16(
            a, b, neg, dt, rows=rows, a_stride=a_stride, b_stride=b_stride
        )

    def chunk_from_conv(self, conv_out, a, b, neg, dt, state, use_qk_l2norm=True):
        return self._mod.gdn_chunk_from_conv_smem_bf16(
            conv_out, a, b, neg, dt, state, use_qk_l2norm=use_qk_l2norm
        )

    def wy_pipeline(self, q16, k16, v48, g, beta, state):
        q16_l2, k16_l2, _, _, g_cumsum = self._mod.gdn_wy_norm_cumsum_pack_qk_bf16(q16, k16, g)
        A = self._mod.gdn_wy_kkt_b64_bf16(k16_l2, beta, g_cumsum)
        Ai = self._mod.gdn_wy_solve_tril_b64_f32(A, q16.shape[0])
        w, u = self._mod.gdn_wy_recompute_wu_b64_bf16(k16_l2, v48, beta, g_cumsum, Ai)
        h0, v_new = self._mod.gdn_wy_chunk_h_b64_bf16(k16_l2, u, w, g_cumsum, state)
        return self._mod.gdn_wy_output_o_b64_bf16(q16_l2, k16_l2, v_new, h0, g_cumsum)


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major == 12 and minor == 1:
        return "12.1"
    if major >= 12:
        return "12.0a"
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "gated_delta_attention_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "gated_delta_attention.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=[
            "-O3",
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
        return InstalledOps(importlib.import_module("gated_delta_attention"))
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_step_inputs(B: int, H: int, seed: int, f32_state=False):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    q = (torch.randn((B, H, D), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    k = (torch.randn((B, H, D), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    v = (torch.randn((B, H, D), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    g = (torch.randn((B, H), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn((B, H), device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    state = (torch.randn((B, H, D, D), device="cuda", generator=gen) * 0.02)
    state = state.float() if f32_state else state.to(torch.bfloat16)
    return q, k, v, g, beta, state


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x.float() * torch.rsqrt((x.float() * x.float()).sum(dim=-1, keepdim=True) + 1e-6)


def ref_recurrent(q, k, v, g, beta, state, *, use_qk_l2norm=True, f32_state=False):
    qs = _norm(q) if use_qk_l2norm else q.float()
    ks = _norm(k) if use_qk_l2norm else k.float()
    qs = qs * (D ** -0.5)
    st = state.float() * torch.exp(g.float())[..., None, None]
    kv_mem = torch.einsum("bhdt,bhd->bht", st, ks)
    delta = (v.float() - kv_mem) * beta.float()[..., None]
    st = st + ks[..., :, None] * delta[..., None, :]
    out = torch.einsum("bhdt,bhd->bht", st, qs).to(torch.bfloat16)
    state_ref = st if f32_state else st.to(torch.bfloat16)
    return out, state_ref


def ref_chunk(q, k, v, g, beta, state, *, use_qk_l2norm=True):
    st = state.unsqueeze(0).clone()
    outs = []
    for i in range(q.shape[0]):
        out, st = ref_recurrent(
            q[i : i + 1].unsqueeze(0).squeeze(1),
            k[i : i + 1].unsqueeze(0).squeeze(1),
            v[i : i + 1].unsqueeze(0).squeeze(1),
            g[i : i + 1].unsqueeze(0).squeeze(1),
            beta[i : i + 1].unsqueeze(0).squeeze(1),
            st,
            use_qk_l2norm=use_qk_l2norm,
        )
        outs.append(out.squeeze(0))
    return torch.stack(outs, dim=0), st.squeeze(0)


def make_conv_inputs(S: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    conv_out = (torch.randn((S, 10240), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    a = (torch.randn((S, 48), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    b = (torch.randn((S, 48), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    neg = (torch.randn((48,), device="cuda", generator=gen).abs() * -0.02).float()
    dt = (torch.randn((48,), device="cuda", generator=gen) * 0.02).float()
    state = (torch.randn((48, D, D), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    return conv_out, a, b, neg, dt, state


def ref_split_broadcast(conv_out):
    S = conv_out.shape[0]
    x = conv_out.view(S, 10240)
    q16 = x[:, :2048].view(S, 16, D)
    k16 = x[:, 2048:4096].view(S, 16, D)
    v48 = x[:, 4096:].view(S, 48, D)
    q48 = q16.repeat_interleave(3, dim=1).contiguous()
    k48 = k16.repeat_interleave(3, dim=1).contiguous()
    return q48, k48, v48.contiguous()


def ref_split_gqa(conv_out):
    S = conv_out.shape[0]
    x = conv_out.view(S, 10240)
    return (
        x[:, :2048].view(S, 16, D).contiguous(),
        x[:, 2048:4096].view(S, 16, D).contiguous(),
        x[:, 4096:].view(S, 48, D).contiguous(),
    )


def ref_gating(a, b, neg, dt):
    g = (neg[None, :] * torch.log1p(torch.exp(a.float() + dt[None, :]))).to(torch.bfloat16)
    beta = torch.sigmoid(b.float()).to(torch.bfloat16)
    return g, beta


def metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (got.float() - ref.float()).abs()
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(torch.quantile(diff.flatten(), 0.99).item()),
        float(torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()),
    )


def run_case(ops, name: str) -> Row:
    kind, B, S, H = SHAPES[name]
    if kind in {"recurrent", "inout", "f32state"}:
        q, k, v, g, beta, state = make_step_inputs(B, H, 7000 + H, f32_state=(kind == "f32state"))
        if kind == "recurrent":
            state_work = state.clone()
            got = ops.recurrent(q, k, v, g, beta, state_work)
            ref, ref_state = ref_recurrent(q, k, v, g, beta, state)
        elif kind == "inout":
            got, state_work = ops.inout(q, k, v, g, beta, state)
            ref, ref_state = ref_recurrent(q, k, v, g, beta, state)
        else:
            state_work = state.clone()
            got = ops.f32state(q, k, v, g, beta, state_work)
            ref, ref_state = ref_recurrent(q, k, v, g, beta, state, f32_state=True)
        torch.cuda.synchronize()
        state_max, _, _, _ = metrics(state_work, ref_state)
        if state_max > (0.00390625 if kind != "f32state" else 0.0005):
            raise AssertionError(f"{name} state mismatch: {state_max}")
    elif kind in {"chunk", "chunk_smem"}:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(9000 + S + H)
        q = (torch.randn((S, H, D), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
        k = (torch.randn((S, H, D), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
        v = (torch.randn((S, H, D), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
        g = (torch.randn((S, H), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
        beta = torch.sigmoid(torch.randn((S, H), device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
        state = (torch.randn((H, D, D), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
        state_work = state.clone()
        got = ops.chunk(q, k, v, g, beta, state_work, smem=(kind == "chunk_smem"))
        ref, ref_state = ref_chunk(q, k, v, g, beta, state)
        torch.cuda.synchronize()
        state_max, _, _, _ = metrics(state_work, ref_state)
        if state_max > 0.00390625:
            raise AssertionError(f"{name} state mismatch: {state_max}")
    if kind == "split":
        conv_out, a, b, neg, dt, state = make_conv_inputs(S, 10000 + S)
        q48, k48, v48 = ops.split_broadcast(conv_out)
        q16, k16, v48_gqa = ops.split_gqa(conv_out)
        q48_ref, k48_ref, v48_ref = ref_split_broadcast(conv_out)
        q16_ref, k16_ref, v48_gqa_ref = ref_split_gqa(conv_out)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(10040 + S)
        q_proj = (torch.randn((S, 24, 512), device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
        q_pre, gate = ops.split_q_gate(q_proj)
        got = torch.cat([
            q48.flatten(), k48.flatten(), v48.flatten(),
            q16.flatten(), k16.flatten(), v48_gqa.flatten(),
            q_pre.flatten(), gate.flatten(),
        ])
        ref = torch.cat([
            q48_ref.flatten(), k48_ref.flatten(), v48_ref.flatten(),
            q16_ref.flatten(), k16_ref.flatten(), v48_gqa_ref.flatten(),
            q_proj[:, :, :256].contiguous().flatten(),
            q_proj[:, :, 256:].contiguous().view(S, 24 * 256).flatten(),
        ])
        max_abs, mean_abs, p99_abs, cos = metrics(got, ref)
        return Row(name, kind, B, S, H, max_abs, mean_abs, p99_abs, cos, max_abs == 0.0)
    if kind == "gating":
        conv_out, a, b, neg, dt, state = make_conv_inputs(S, 11000 + S)
        g, beta = ops.gating(a, b, neg, dt)
        g_ref, beta_ref = ref_gating(a, b, neg, dt)
        a_pad = torch.empty((S, 64), device="cuda", dtype=torch.bfloat16)
        b_pad = torch.empty_like(a_pad)
        a_pad[:, :48] = a
        b_pad[:, :48] = b
        gs, betas = ops.gating_strided(a_pad.flatten(), b_pad.flatten(), neg, dt, S, 64, 64)
        got = torch.cat([g.flatten(), beta.flatten(), gs.flatten(), betas.flatten()])
        ref = torch.cat([g_ref.flatten(), beta_ref.flatten(), g_ref.flatten(), beta_ref.flatten()])
        max_abs, mean_abs, p99_abs, cos = metrics(got, ref)
        return Row(name, kind, B, S, H, max_abs, mean_abs, p99_abs, cos, max_abs <= 0.001953125 and cos >= 0.9999)
    if kind == "chunk_from_conv":
        conv_out, a, b, neg, dt, state = make_conv_inputs(S, 12000 + S)
        state_work = state.clone()
        got = ops.chunk_from_conv(conv_out, a, b, neg, dt, state_work)
        q48, k48, v48 = ref_split_broadcast(conv_out)
        g, beta = ref_gating(a, b, neg, dt)
        ref, ref_state = ref_chunk(q48, k48, v48, g, beta, state)
        state_max, _, _, _ = metrics(state_work, ref_state)
        if state_max > 0.00390625:
            raise AssertionError(f"{name} state mismatch: {state_max}")
    if kind == "wy_pipeline":
        conv_out, a, b, neg, dt, state = make_conv_inputs(S, 13000 + S)
        q16, k16, v48 = ref_split_gqa(conv_out)
        g, beta = ref_gating(a, b, neg, dt)
        state_work = state.clone()
        got = ops.wy_pipeline(q16, k16, v48, g, beta, state_work)
        q48 = q16.repeat_interleave(3, dim=1).contiguous()
        k48 = k16.repeat_interleave(3, dim=1).contiguous()
        ref, ref_state = ref_chunk(q48, k48, v48, g, beta, state)
        state_max, _, _, _ = metrics(state_work, ref_state)
        if state_max > 0.015625:
            raise AssertionError(f"{name} state mismatch: {state_max}")
    max_abs, mean_abs, p99_abs, cos = metrics(got, ref)
    passed = max_abs <= 0.015625 and mean_abs <= 0.0015 and cos >= 0.999
    return Row(name, kind, B, S, H, max_abs, mean_abs, p99_abs, cos, passed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=sorted(MODES), default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    rows = []
    for name in MODES[args.mode]:
        row = run_case(ops, name)
        rows.append(row)
        print(
            f"{row.name}: max_abs={row.max_abs:.6f} mean_abs={row.mean_abs:.6f} "
            f"p99_abs={row.p99_abs:.6f} cosine={row.cosine:.8f} passed={row.passed}"
        )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n")
    if not all(r.passed for r in rows):
        raise AssertionError("gated-delta-attention correctness failed")
    print(f"PASS gated-delta-attention {args.backend} mode={args.mode}: {len(rows)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
