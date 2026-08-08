#!/usr/bin/env python3
"""Correctness tests for transformer-fused-ops."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "transformer-fused-ops"
REGISTRATION_INCLUDE = ROOT.parent / "kernels" / "kernel-builder" / "src" / "pyproject" / "templates" / "torch"


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self.ops = getattr(torch.ops, namespace)

    def rms_norm_gated_silu_bf16(self, x, gate, weight, eps=1e-6):
        out = torch.empty_like(x)
        self.ops.rms_norm_gated_silu_bf16(x, gate, weight, float(eps), out)
        return out

    def silu_mul_bf16(self, gate, up):
        out = torch.empty_like(gate)
        self.ops.silu_mul_bf16(gate, up, out)
        return out

    def sigmoid_mul_bf16(self, gate, x):
        out = torch.empty_like(gate)
        self.ops.sigmoid_mul_bf16(gate, x, out)
        return out

    def embedding_lookup_bf16(self, token_ids, embed):
        out = torch.empty((token_ids.shape[0], embed.shape[1]), device=embed.device, dtype=torch.bfloat16)
        self.ops.embedding_lookup_bf16(token_ids, embed, out)
        return out

    def partial_rope_qk_bf16(self, q, k, cos, sin, rope_dim):
        qo = torch.empty_like(q)
        ko = torch.empty_like(k)
        self.ops.partial_rope_qk_bf16(q, k, cos, sin, qo, ko, int(rope_dim))
        return qo, ko

    def argmax_bf16(self, logits):
        out = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
        self.ops.argmax_bf16(logits, out)
        return out

    def spec_accept_greedy_bf16(self, logits, drafts, spec_k):
        argmax = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
        accept_n = torch.empty((1,), device=logits.device, dtype=torch.int32)
        self.ops.spec_accept_greedy_bf16(logits, drafts, argmax, accept_n, int(spec_k))
        return argmax, accept_n

    def nexn2_lin_split_qkv_broadcast_bf16(self, conv_out):
        q = torch.empty((conv_out.shape[0], 32, 128), device=conv_out.device, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        self.ops.nexn2_lin_split_qkv_broadcast_bf16(conv_out, q, k, v)
        return q, k, v

    def nexn2_split_q_gate_bf16(self, q_proj):
        q_pre = torch.empty((q_proj.shape[0], 16, 256), device=q_proj.device, dtype=torch.bfloat16)
        gate = torch.empty((q_proj.shape[0], 16 * 256), device=q_proj.device, dtype=torch.bfloat16)
        self.ops.nexn2_split_q_gate_bf16(q_proj, q_pre, gate)
        return q_pre, gate

    def nexn2_router_topk_bf16(self, logits, k=8):
        idx = torch.empty((k,), device=logits.device, dtype=torch.int32)
        val = torch.empty((k,), device=logits.device, dtype=torch.float32)
        self.ops.nexn2_router_topk_bf16(logits, idx, val, int(k))
        return idx, val

    def router_topk_bf16(self, logits, k=8):
        idx = torch.empty((k,), device=logits.device, dtype=torch.int32)
        val = torch.empty((k,), device=logits.device, dtype=torch.float32)
        self.ops.router_topk_bf16(logits, idx, val, int(k))
        return idx, val

    def moe_weighted_sum_bf16_to_fp32(
        self, expert_output, row_indices, router_weight, hidden=None, out=None
    ):
        hidden = expert_output.shape[1] if hidden is None else int(hidden)
        if out is None:
            out = torch.empty(
                (row_indices.shape[0], hidden),
                device=expert_output.device,
                dtype=torch.float32,
            )
        self.ops.moe_weighted_sum_bf16_to_fp32(expert_output, row_indices, router_weight, out)
        return out

    def relu2_quantize_fp8_static_bf16(self, x, scale):
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self.ops.relu2_quantize_fp8_static_bf16(x, scale, out)
        return out

    def rms_norm_fp16(self, x, weight, eps=1e-6):
        out = torch.empty_like(x)
        self.ops.rms_norm_fp16(x, weight, float(eps), out)
        return out

    def layer_norm_fp16(self, x, weight, bias, eps=1e-6):
        out = torch.empty_like(x)
        self.ops.layer_norm_fp16(x, weight, bias, float(eps), out)
        return out

    def layer_norm_quant_fp8_static_fp16(self, x, weight, bias, scale, eps=1e-6):
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self.ops.layer_norm_quant_fp8_static_fp16(
            x, weight, bias, scale, float(eps), out
        )
        return out

    def rope_rotate_half_fp16_(self, x, cos, sin):
        self.ops.rope_rotate_half_fp16_(x, cos, sin)
        return x

    def quantize_fp8_static_fp16(self, x, scale):
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self.ops.quantize_fp8_static_fp16(x, scale, out)
        return out

    def quantize_fp8_static_bf16(self, x, scale, out=None):
        if out is None:
            out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self.ops.quantize_fp8_static_bf16(x, scale, out)
        return out

    def layer_norm_quant_fp8_static_bf16(
        self, x, weight, bias, scale, eps=1e-6, out=None
    ):
        if out is None:
            out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        self.ops.layer_norm_quant_fp8_static_bf16(
            x, weight, bias, scale, float(eps), out
        )
        return out

    def gate_geglu_merged_quant_fp8_static_bf16(
        self, merged, scale, out=None
    ):
        if out is None:
            out = torch.empty(
                (merged.shape[0], merged.shape[1] // 2),
                device=merged.device,
                dtype=torch.float8_e4m3fn,
            )
        self.ops.gate_geglu_merged_quant_fp8_static_bf16(merged, scale, out)
        return out

    def residual_add_fp16_(self, residual, x):
        self.ops.residual_add_fp16_(residual, x)
        return residual

    def repeat_interleave_heads_fp16(self, x, repeat):
        out = torch.empty(
            (x.shape[0], x.shape[1] * repeat, x.shape[2]),
            device=x.device,
            dtype=x.dtype,
        )
        self.ops.repeat_interleave_heads_fp16(x, int(repeat), out)
        return out


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return "12.0a" if major >= 12 else f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "transformer_fused_ops_source_test"
    sources = [
        str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
        str(PACKAGE / "csrc" / "kernels" / "rms_norm_gated_silu_qwen36.cu"),
        str(PACKAGE / "csrc" / "kernels" / "silu_mul_qwen36.cu"),
        str(PACKAGE / "csrc" / "kernels" / "qwen36_misc.cu"),
        str(PACKAGE / "csrc" / "kernels" / "nexn2_misc.cu"),
        str(PACKAGE / "csrc" / "kernels" / "nexn2_router_topk.cu"),
        str(PACKAGE / "csrc" / "kernels" / "moe_weighted_sum_sm120.cu"),
        str(PACKAGE / "csrc" / "kernels" / "relu2_quantize_fp8.cu"),
    ]
    if torch.cuda.get_device_capability(0) == (11, 0):
        sources.extend([
            str(PACKAGE / "csrc" / "kernels" / "vec_fp16_backbone.cu"),
            str(PACKAGE / "csrc" / "kernels" / "vec_bf16_producers.cu"),
            str(PACKAGE / "csrc" / "kernels" / "vec_fp16_dispatch.cu"),
        ])
    load(
        name=namespace,
        sources=sources,
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        is_python_module=False,
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("transformer_fused_ops")
    finally:
        if artifact:
            sys.path.remove(artifact)


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float = 0.00390625) -> None:
    diff = (got.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    cos = float(torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item())
    if max_abs > atol or cos < 0.999:
        raise AssertionError(f"{name}: max_abs={max_abs:.8f} cos={cos:.8f}")


def assert_fp8_distribution(
    name: str, got: torch.Tensor, ref: torch.Tensor, max_mismatch_fraction: float
) -> None:
    got_f = got.float()
    ref_f = ref.float()
    diff = (got_f - ref_f).abs().flatten()
    mismatch = int((diff != 0).sum().item())
    fraction = mismatch / diff.numel()
    p99 = float(torch.quantile(diff, 0.99).item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            got_f.flatten(), ref_f.flatten(), dim=0
        ).item()
    )
    if p99 != 0.0 or fraction > max_mismatch_fraction or cosine < 0.9999:
        raise AssertionError(
            f"{name}: mismatch={mismatch}/{diff.numel()} fraction={fraction:.8f} "
            f"p99={p99:.8f} max={float(diff.max().item()):.8f} cosine={cosine:.8f}"
        )


def rope_ref(x, cos, sin, rope_dim):
    out = x.clone()
    half = rope_dim // 2
    left = x[:, :, :half].float()
    right = x[:, :, half:rope_dim].float()
    out[:, :, :half] = ((-right * sin[:, None, :half].float()).to(torch.bfloat16).float() + left * cos[:, None, :half].float()).to(torch.bfloat16)
    out[:, :, half:rope_dim] = ((left * sin[:, None, half:rope_dim].float()).to(torch.bfloat16).float() + right * cos[:, None, half:rope_dim].float()).to(torch.bfloat16)
    return out


def run(ops, mode: str) -> int:
    rows = [1, 8] if mode == "smoke" else [1, 8, 64, 257]
    count = 0
    for m in rows:
        x = (torch.randn((m, 128), device="cuda") * 0.1).to(torch.bfloat16)
        gate = (torch.randn_like(x.float()) * 0.1).to(torch.bfloat16)
        w = torch.randn((128,), device="cuda").to(torch.bfloat16)
        got = ops.rms_norm_gated_silu_bf16(x, gate, w)
        norm = x.float() * torch.rsqrt((x.float() * x.float()).mean(dim=1, keepdim=True) + 1e-6)
        weighted = (w.float() * norm.to(torch.bfloat16).float()).to(torch.bfloat16)
        ref = (weighted.float() * torch.nn.functional.silu(gate.float())).to(torch.bfloat16)
        assert_close(f"rms_norm_gated_silu rows={m}", got, ref, 0.00390625)
        count += 1

    gate = (torch.randn((4, 1024), device="cuda") * 0.2).to(torch.bfloat16)
    up = (torch.randn_like(gate.float()) * 0.2).to(torch.bfloat16)
    assert_close("silu_mul", ops.silu_mul_bf16(gate, up), (torch.nn.functional.silu(gate.float()).to(torch.bfloat16).float() * up.float()).to(torch.bfloat16))
    assert_close("sigmoid_mul", ops.sigmoid_mul_bf16(gate, up), (torch.sigmoid(gate.float()).to(torch.bfloat16).float() * up.float()).to(torch.bfloat16))
    count += 2

    token_ids = torch.tensor([0, 3, 7, 11], device="cuda", dtype=torch.int64)
    embed = torch.arange(16 * 32, device="cuda", dtype=torch.float32).reshape(16, 32).to(torch.bfloat16)
    got = ops.embedding_lookup_bf16(token_ids, embed)
    if not torch.equal(got.cpu(), embed[token_ids].cpu()):
        raise AssertionError("embedding lookup mismatch")
    count += 1

    if torch.cuda.get_device_capability(0) == (11, 0):
        for shape in ((1, 127), (51, 1536), (712, 2048), (768, 4304)):
            x_bf16 = (torch.randn(shape, device="cuda") * 0.5).to(torch.bfloat16)
            scale_bf16 = torch.tensor([0.025], device="cuda", dtype=torch.float32)
            quantized = ops.quantize_fp8_static_bf16(x_bf16, scale_bf16)
            quantized_ref = (
                x_bf16.float().div(scale_bf16).clamp(-448.0, 448.0)
            ).to(torch.float8_e4m3fn)
            if not torch.equal(quantized, quantized_ref):
                raise AssertionError(f"quantize_fp8_static_bf16 mismatch for {shape}")
            count += 1

        for norm_rows, dim in ((512, 1152), (712, 2048), (768, 4304)):
            x_bf16 = torch.randn(
                (norm_rows, dim), device="cuda", dtype=torch.bfloat16
            )
            weight_bf16 = torch.randn(
                (dim,), device="cuda", dtype=torch.bfloat16
            )
            bias_bf16 = torch.randn(
                (dim,), device="cuda", dtype=torch.bfloat16
            )
            scale_bf16 = torch.tensor(
                [0.04], device="cuda", dtype=torch.float32
            )
            fused_bf16 = ops.layer_norm_quant_fp8_static_bf16(
                x_bf16, weight_bf16, bias_bf16, scale_bf16
            )
            norm_ref = torch.nn.functional.layer_norm(
                x_bf16.float(), (dim,), weight_bf16.float(),
                bias_bf16.float(), 1e-6
            ).to(torch.bfloat16)
            fused_ref = (
                norm_ref.float().div(scale_bf16).clamp(-448.0, 448.0)
            ).to(torch.float8_e4m3fn)
            assert_fp8_distribution(
                f"layer_norm_quant_fp8_static_bf16_{norm_rows}_{dim}",
                fused_bf16,
                fused_ref,
                0.002,
            )
            count += 1

        for geglu_rows, hidden in ((51, 4096), (512, 4304), (768, 3456)):
            merged = (torch.randn(
                (geglu_rows, 2 * hidden), device="cuda"
            ) * 0.25).to(torch.bfloat16)
            scale_bf16 = torch.tensor(
                [0.025], device="cuda", dtype=torch.float32
            )
            geglu = ops.gate_geglu_merged_quant_fp8_static_bf16(
                merged, scale_bf16
            )
            gate, up = merged.float().chunk(2, dim=-1)
            geglu_ref = (
                (
                    gate
                    / (
                        1.0
                        + torch.exp(
                            -1.5957691216057308
                            * gate
                            * (1.0 + 0.044715 * gate.square())
                        )
                    )
                    * up
                )
                .div(scale_bf16)
                .clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
            )
            assert_fp8_distribution(
                f"gate_geglu_merged_quant_fp8_static_bf16_{geglu_rows}_{hidden}",
                geglu,
                geglu_ref,
                0.001,
            )
            count += 1

        for norm_rows, dim in ((277 * 16, 128), (277, 2048), (1024, 1024)):
            x16 = torch.randn((norm_rows, dim), device="cuda", dtype=torch.float16)
            w16 = torch.randn((dim,), device="cuda", dtype=torch.float16)
            b16 = torch.randn((dim,), device="cuda", dtype=torch.float16)
            rms = ops.rms_norm_fp16(x16, w16)
            rms_ref = (
                x16.float()
                * torch.rsqrt(x16.float().square().mean(-1, keepdim=True) + 1e-6)
                * w16.float()
            ).half()
            assert_close(f"rms_norm_fp16_{norm_rows}_{dim}", rms, rms_ref, 0.0078125)
            ln = ops.layer_norm_fp16(x16, w16, b16)
            ln_ref = torch.nn.functional.layer_norm(
                x16.float(), (dim,), w16.float(), b16.float(), 1e-6
            ).half()
            assert_close(f"layer_norm_fp16_{norm_rows}_{dim}", ln, ln_ref, 0.015625)
            scale16 = torch.tensor([0.01], device="cuda", dtype=torch.float32)
            ln_fp8 = ops.layer_norm_quant_fp8_static_fp16(
                x16, w16, b16, scale16
            )
            staged_fp8 = ops.quantize_fp8_static_fp16(ln, scale16)
            if not torch.equal(ln_fp8.view(torch.uint8), staged_fp8.view(torch.uint8)):
                raise AssertionError("fused LayerNorm-FP8 differs from staged native ops")
            count += 3

        sequence, heads, head_dim = 277, 16, 128
        rope_x = torch.randn(
            (sequence, heads, head_dim), device="cuda", dtype=torch.float16
        )
        cos16 = torch.randn(
            (sequence, head_dim), device="cuda", dtype=torch.float16
        )
        sin16 = torch.randn_like(cos16)
        rope_ref_x = rope_x.clone()
        half = head_dim // 2
        left = rope_ref_x[..., :half].float()
        right = rope_ref_x[..., half:].float()
        rope_expected = torch.empty_like(rope_ref_x)
        rope_expected[..., :half] = (
            left * cos16[:, None, :half].float()
            - right * sin16[:, None, :half].float()
        ).half()
        rope_expected[..., half:] = (
            right * cos16[:, None, :half].float()
            + left * sin16[:, None, :half].float()
        ).half()
        got_rope = ops.rope_rotate_half_fp16_(rope_x.clone(), cos16, sin16)
        assert_close("rope_rotate_half_fp16", got_rope, rope_expected, 0.00390625)

        repeat_src = torch.randn(
            (277, 8, 128), device="cuda", dtype=torch.float16
        )
        repeated = ops.repeat_interleave_heads_fp16(repeat_src, 2)
        if not torch.equal(repeated, repeat_src.repeat_interleave(2, dim=1)):
            raise AssertionError("repeat_interleave_heads_fp16 mismatch")
        residual = torch.randn((41, 1536), device="cuda", dtype=torch.float16)
        update = torch.randn_like(residual)
        expected_residual = (residual.float() + update.float()).half()
        got_residual = ops.residual_add_fp16_(residual.clone(), update)
        if not torch.equal(got_residual, expected_residual):
            raise AssertionError("residual_add_fp16_ mismatch")
        count += 3

        compile_x_bf16 = torch.randn(
            (51, 1536), device="cuda", dtype=torch.bfloat16
        )
        compile_scale_bf16 = torch.tensor(
            [0.025], device="cuda", dtype=torch.float32
        )

        def invoke_bf16(value, scale):
            return ops.quantize_fp8_static_bf16(value, scale)

        eager_bf16 = invoke_bf16(compile_x_bf16, compile_scale_bf16)
        compiled_bf16 = torch.compile(invoke_bf16, fullgraph=True)(
            compile_x_bf16, compile_scale_bf16
        )
        if not torch.equal(compiled_bf16, eager_bf16):
            raise AssertionError("BF16 quantize compile mismatch")

        graph_out = torch.empty_like(
            compile_x_bf16, dtype=torch.float8_e4m3fn
        )
        ops.quantize_fp8_static_bf16(compile_x_bf16, compile_scale_bf16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ops.quantize_fp8_static_bf16(
                compile_x_bf16, compile_scale_bf16, out=graph_out
            )
        graph.replay()
        if not torch.equal(graph_out, eager_bf16):
            raise AssertionError("BF16 quantize CUDA Graph mismatch")

        compile_norm_x = torch.randn(
            (51, 128), device="cuda", dtype=torch.bfloat16
        )
        compile_norm_w = torch.randn(
            (128,), device="cuda", dtype=torch.bfloat16
        )
        compile_norm_b = torch.randn_like(compile_norm_w)

        def invoke_ln(value, weight, bias, scale):
            return ops.layer_norm_quant_fp8_static_bf16(
                value, weight, bias, scale
            )

        eager_ln = invoke_ln(
            compile_norm_x, compile_norm_w, compile_norm_b,
            compile_scale_bf16
        )
        compiled_ln = torch.compile(invoke_ln, fullgraph=True)(
            compile_norm_x, compile_norm_w, compile_norm_b,
            compile_scale_bf16
        )
        if not torch.equal(compiled_ln, eager_ln):
            raise AssertionError("BF16 LayerNorm producer compile mismatch")
        graph_ln = torch.empty_like(eager_ln)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ops.layer_norm_quant_fp8_static_bf16(
                compile_norm_x, compile_norm_w, compile_norm_b,
                compile_scale_bf16, out=graph_ln
            )
        graph.replay()
        if not torch.equal(graph_ln, eager_ln):
            raise AssertionError("BF16 LayerNorm producer CUDA Graph mismatch")

        compile_merged = torch.randn(
            (51, 256), device="cuda", dtype=torch.bfloat16
        )

        def invoke_geglu(value, scale):
            return ops.gate_geglu_merged_quant_fp8_static_bf16(value, scale)

        eager_geglu = invoke_geglu(compile_merged, compile_scale_bf16)
        compiled_geglu = torch.compile(invoke_geglu, fullgraph=True)(
            compile_merged, compile_scale_bf16
        )
        if not torch.equal(compiled_geglu, eager_geglu):
            raise AssertionError("BF16 GeGLU producer compile mismatch")
        graph_geglu = torch.empty_like(eager_geglu)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ops.gate_geglu_merged_quant_fp8_static_bf16(
                compile_merged, compile_scale_bf16, out=graph_geglu
            )
        graph.replay()
        if not torch.equal(graph_geglu, eager_geglu):
            raise AssertionError("BF16 GeGLU producer CUDA Graph mismatch")
        count += 6

    q = torch.randn((8, 4, 128), device="cuda").to(torch.bfloat16)
    k = torch.randn((8, 2, 128), device="cuda").to(torch.bfloat16)
    cos = torch.randn((8, 64), device="cuda").to(torch.bfloat16)
    sin = torch.randn((8, 64), device="cuda").to(torch.bfloat16)
    qg, kg = ops.partial_rope_qk_bf16(q, k, cos, sin, 64)
    assert_close("partial_rope_q", qg, rope_ref(q, cos, sin, 64))
    assert_close("partial_rope_k", kg, rope_ref(k, cos, sin, 64))
    count += 2

    logits = torch.randn((5, 1024), device="cuda").to(torch.bfloat16)
    got = ops.argmax_bf16(logits)
    if not torch.equal(got.cpu(), torch.argmax(logits.float(), dim=1).cpu()):
        raise AssertionError("argmax mismatch")
    drafts = got.clone()
    drafts[3:] += 1
    _, accept_n = ops.spec_accept_greedy_bf16(logits, drafts, 5)
    torch.cuda.synchronize()
    if int(accept_n.cpu()[0]) != 3:
        raise AssertionError("spec accept mismatch")
    count += 2

    conv = torch.randn((3, 8192), device="cuda").to(torch.bfloat16)
    q32, k32, v32 = ops.nexn2_lin_split_qkv_broadcast_bf16(conv)
    ref_q = conv[:, :2048].reshape(3, 16, 128)[:, torch.arange(32, device="cuda") // 2]
    ref_k = conv[:, 2048:4096].reshape(3, 16, 128)[:, torch.arange(32, device="cuda") // 2]
    ref_v = conv[:, 4096:].reshape(3, 32, 128)
    assert_close("nexn2_lin_q", q32, ref_q)
    assert_close("nexn2_lin_k", k32, ref_k)
    assert_close("nexn2_lin_v", v32, ref_v)
    count += 3

    q_proj = torch.randn((3, 16, 512), device="cuda").to(torch.bfloat16)
    q_pre, q_gate = ops.nexn2_split_q_gate_bf16(q_proj)
    assert_close("nexn2_q_pre", q_pre, q_proj[:, :, :256].contiguous())
    assert_close("nexn2_gate", q_gate, q_proj[:, :, 256:].reshape(3, 16 * 256).contiguous())
    count += 2

    router = torch.linspace(-1.0, 1.0, 256, device="cuda").to(torch.bfloat16)
    idx, val = ops.nexn2_router_topk_bf16(router, 8)
    ref_val, ref_idx = torch.topk(router.float(), 8)
    if not torch.equal(idx.cpu(), ref_idx.to(torch.int32).cpu()) or not torch.allclose(val.cpu(), ref_val.cpu()):
        raise AssertionError("router topk mismatch")
    count += 1

    generic_idx, generic_val = ops.router_topk_bf16(router, 8)
    if not torch.equal(generic_idx, idx) or not torch.equal(generic_val, val):
        raise AssertionError("generic router alias mismatch")
    count += 1

    for tokens, topk, hidden, stride in [(1, 1, 128, 128), (3, 4, 320, 384), (17, 8, 2048, 2112)]:
        routed_rows = tokens * topk + 5
        expert_output = torch.randn((routed_rows, stride), device="cuda", dtype=torch.bfloat16)
        row_indices = torch.randint(0, routed_rows, (tokens, topk), device="cuda", dtype=torch.int32)
        router_weight = torch.softmax(torch.randn((tokens, topk), device="cuda"), dim=-1)
        got = ops.moe_weighted_sum_bf16_to_fp32(
            expert_output, row_indices, router_weight, hidden=hidden
        )
        ref = (
            expert_output[row_indices.long(), :hidden].float()
            * router_weight[..., None]
        ).sum(dim=1)
        assert_close(f"moe_weighted_sum_{tokens}_{topk}_{hidden}", got, ref, 1e-5)
        count += 1

    for shape in [(1, 128), (51, 1536), (277, 2048), (1024, 4096)]:
        x = (torch.randn(shape, device="cuda") * 0.5).to(torch.bfloat16)
        scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)
        got = ops.relu2_quantize_fp8_static_bf16(x, scale)
        expected = (
            torch.relu(x.float()).square().div(scale).clamp(max=448.0)
        ).to(torch.float8_e4m3fn)
        if not torch.equal(got, expected):
            mismatch = int((got != expected).sum().item())
            raise AssertionError(f"relu2 quant mismatch: {mismatch}")
        count += 1

    compile_x = torch.randn(
        (51, 1536), device="cuda", dtype=torch.bfloat16
    )
    compile_scale = torch.tensor(
        [0.01], device="cuda", dtype=torch.float32
    )

    def invoke(value, scale):
        return ops.relu2_quantize_fp8_static_bf16(value, scale)

    eager = invoke(compile_x, compile_scale)
    compiled = torch.compile(invoke, fullgraph=True)(
        compile_x, compile_scale
    )
    if not torch.equal(compiled, eager):
        raise AssertionError("relu2 quant compile mismatch")
    count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    count = run(ops, args.mode)
    print(f"transformer-fused-ops {args.backend} {args.mode}: passed {count}/{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
