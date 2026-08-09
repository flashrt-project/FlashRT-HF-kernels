#!/usr/bin/env python3
"""Benchmark the H32 WY path against the existing H48 native specialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def time_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def make_workspace(S: int, Hv: int, Hk: int, D: int, device: str):
    C = (S + 63) // 64
    bf16 = torch.bfloat16
    return {
        "q_l2": torch.empty((S, Hk, D), device=device, dtype=bf16),
        "k_l2": torch.empty((S, Hk, D), device=device, dtype=bf16),
        "q_pack": torch.empty((C, Hv, 64, D), device=device, dtype=bf16),
        "k_pack": torch.empty((C, Hk, 64, D), device=device, dtype=bf16),
        "g_cumsum": torch.empty((S, Hv), device=device, dtype=bf16),
        "A": torch.empty((C, Hv, 64, 64), device=device, dtype=torch.float32),
        "Ai": torch.empty((C, Hv, 64, 64), device=device, dtype=torch.float32),
        "Ai_pack": torch.empty((C, Hv, 64, 64), device=device, dtype=bf16),
        "w_pack": torch.empty((C, Hv, 64, D), device=device, dtype=bf16),
        "u_pack": torch.empty((C, Hv, 64, D), device=device, dtype=bf16),
        "h0": torch.empty((C, Hv, D, D), device=device, dtype=bf16),
        "v_new": torch.empty((S, Hv, D), device=device, dtype=bf16),
        "v_new_pack": torch.empty((C, Hv, 64, D), device=device, dtype=bf16),
        "k_pack_hv": torch.empty((C, Hv, 64, D), device=device, dtype=bf16),
        "out": torch.empty((S, Hv, D), device=device, dtype=bf16),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("source", "installed"), default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--sequences", default="64,65,128,256")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    tests = Path(__file__).resolve().parents[1] / "tests"
    sys.path.insert(0, str(tests))
    from test_gated_delta_attention import load_installed_ops, load_source_ops

    if args.backend == "source":
        raw = load_source_ops()._ops
    else:
        if not args.artifact:
            raise ValueError("--artifact is required for installed backend")
        raw = load_installed_ops(args.artifact)._mod.ops
    D, Hk = 128, 16
    results = []
    for S in (int(v) for v in args.sequences.split(",")):
        rows = {}
        for Hv in (32, 48):
            gen = torch.Generator(device="cuda").manual_seed(810000 + S + Hv)
            q = (torch.randn((S, Hk, D), device="cuda", generator=gen) * 0.05).bfloat16()
            k = (torch.randn((S, Hk, D), device="cuda", generator=gen) * 0.05).bfloat16()
            v = (torch.randn((S, Hv, D), device="cuda", generator=gen) * 0.05).bfloat16()
            g = (torch.randn((S, Hv), device="cuda", generator=gen) * 0.02).bfloat16()
            beta = torch.sigmoid(torch.randn((S, Hv), device="cuda", generator=gen)).bfloat16()
            initial = (torch.randn((Hv, D, D), device="cuda", generator=gen) * 0.02).bfloat16()
            state = initial.clone()
            ws = make_workspace(S, Hv, Hk, D, "cuda")

            def launch():
                state.copy_(initial)
                if Hv == 48:
                    raw.gdn_wy_norm_cumsum_pack_qk_bf16(q, k, g, ws["q_l2"], ws["k_l2"], ws["q_pack"], ws["k_pack"], ws["g_cumsum"])
                    raw.gdn_wy_kkt_b64_bf16(ws["k_l2"], beta, ws["g_cumsum"], ws["A"])
                    raw.gdn_wy_solve_tril_b64_f32(ws["A"], ws["Ai"], S)
                    raw.gdn_wy_cast_ai_f32_to_bf16(ws["Ai"], ws["Ai_pack"], S)
                    raw.gdn_wy_recompute_wu_b64_mma_fla_bf16(ws["k_l2"], v, beta, ws["g_cumsum"], ws["Ai_pack"], ws["w_pack"], ws["u_pack"])
                    raw.gdn_wy_chunk_h_b64_mma_fla_bf16(ws["k_l2"], ws["w_pack"], ws["u_pack"], ws["g_cumsum"], state, ws["h0"], ws["v_new"], ws["v_new_pack"], ws["k_pack_hv"])
                    raw.gdn_wy_output_o_b64_mma_fla_bf16(ws["q_pack"], ws["k_pack_hv"], ws["v_new_pack"], ws["h0"], ws["g_cumsum"], ws["out"], D ** -0.5)
                else:
                    raw.gdn_wy_norm_cumsum_pack_qk_h_bf16(q, k, g, ws["q_l2"], ws["k_l2"], ws["q_pack"], ws["k_pack"], ws["g_cumsum"], Hv, Hk, D)
                    raw.gdn_wy_kkt_b64_h_bf16(ws["k_l2"], beta, ws["g_cumsum"], ws["A"], Hv, Hk, D)
                    raw.gdn_wy_solve_tril_b64_h_f32(ws["A"], ws["Ai"], S, Hv)
                    raw.gdn_wy_cast_ai_h_f32_to_bf16(ws["Ai"], ws["Ai_pack"], S, Hv)
                    raw.gdn_wy_recompute_wu_b64_mma_fla_h_bf16(ws["k_l2"], v, beta, ws["g_cumsum"], ws["Ai_pack"], ws["w_pack"], ws["u_pack"], Hv, Hk, D)
                    raw.gdn_wy_chunk_h_b64_mma_fla_h_bf16(ws["k_l2"], ws["w_pack"], ws["u_pack"], ws["g_cumsum"], state, ws["h0"], ws["v_new"], ws["v_new_pack"], ws["k_pack_hv"], Hv, Hk, D)
                    raw.gdn_wy_output_o_b64_mma_fla_h_bf16(ws["q_pack"], ws["k_pack_hv"], ws["v_new_pack"], ws["h0"], ws["g_cumsum"], ws["out"], Hv, Hk, D, D ** -0.5)
                return ws["out"]

            rows[Hv] = time_us(launch, args.warmup, args.iterations)
        ratio = rows[32] / rows[48]
        per_head = (rows[32] / 32) / (rows[48] / 48)
        results.append({
            "sequence": S,
            "h32_us": rows[32],
            "h48_native_us": rows[48],
            "h32_h48_ratio": ratio,
            "h32_h48_per_head_ratio": per_head,
        })
        print(f"S={S} h32_us={rows[32]:.3f} h48_native_us={rows[48]:.3f} raw_ratio={ratio:.3f} per_head_ratio={per_head:.3f}")
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
