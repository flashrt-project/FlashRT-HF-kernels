#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))
import flashrt_flex_attention_train as flex_ops  # noqa: E402


FLEX_TILE_PRESETS = {
    "default": None,
    # fwd fully autotuned; backward constrained to the consumer-GPU tiles
    # (autotuned backward at GQA/D=256 exceeds 5090 shared memory).
    "bwd_shrunk_only": {
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 64,
        "bwd_BLOCK_M2": 64,
        "bwd_BLOCK_N2": 32,
    },
    "torch_default_explicit": {
        "fwd_BLOCK_M": 32,
        "fwd_BLOCK_N": 64,
        "fwd_num_stages": 2,
        "fwd_num_warps": 4,
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 64,
        "bwd_BLOCK_M2": 64,
        "bwd_BLOCK_N2": 32,
        "bwd_num_stages": 1,
        "bwd_num_warps": 4,
    },
    "a100_d256_bwd_32x64": {
        "fwd_BLOCK_M": 32,
        "fwd_BLOCK_N": 64,
        "fwd_num_stages": 2,
        "fwd_num_warps": 4,
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 64,
        "bwd_BLOCK_M2": 64,
        "bwd_BLOCK_N2": 32,
        "bwd_num_stages": 3,
        "bwd_num_warps": 4,
    },
    "a100_d256_bwd_32x128": {
        "fwd_BLOCK_M": 32,
        "fwd_BLOCK_N": 64,
        "fwd_num_stages": 2,
        "fwd_num_warps": 4,
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 128,
        "bwd_BLOCK_M2": 128,
        "bwd_BLOCK_N2": 32,
        "bwd_num_stages": 3,
        "bwd_num_warps": 8,
    },
    "a100_d256_bwd_64x64": {
        "fwd_BLOCK_M": 64,
        "fwd_BLOCK_N": 64,
        "fwd_num_stages": 3,
        "fwd_num_warps": 4,
        "bwd_BLOCK_M1": 64,
        "bwd_BLOCK_N1": 64,
        "bwd_BLOCK_M2": 64,
        "bwd_BLOCK_N2": 64,
        "bwd_num_stages": 3,
        "bwd_num_warps": 4,
    },
    "a100_d256_bwd_64x128": {
        "fwd_BLOCK_M": 64,
        "fwd_BLOCK_N": 128,
        "fwd_num_stages": 3,
        "fwd_num_warps": 4,
        "bwd_BLOCK_M1": 64,
        "bwd_BLOCK_N1": 128,
        "bwd_BLOCK_M2": 128,
        "bwd_BLOCK_N2": 64,
        "bwd_num_stages": 3,
        "bwd_num_warps": 8,
    },
    "a100_d256_bwd_write_dq_false": {
        "fwd_BLOCK_M": 32,
        "fwd_BLOCK_N": 64,
        "fwd_num_stages": 2,
        "fwd_num_warps": 4,
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 64,
        "bwd_BLOCK_M2": 64,
        "bwd_BLOCK_N2": 32,
        "bwd_num_stages": 3,
        "bwd_num_warps": 4,
        "WRITE_DQ": False,
    },
    "a100_d256_prescale_safe": {
        "fwd_BLOCK_M": 32,
        "fwd_BLOCK_N": 64,
        "fwd_num_stages": 2,
        "fwd_num_warps": 4,
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 64,
        "bwd_BLOCK_M2": 64,
        "bwd_BLOCK_N2": 32,
        "bwd_num_stages": 3,
        "bwd_num_warps": 4,
        "PRESCALE_QK": True,
        "ROWS_GUARANTEED_SAFE": True,
    },
    "a100_d256_contig_safe": {
        "fwd_BLOCK_M": 32,
        "fwd_BLOCK_N": 64,
        "fwd_num_stages": 2,
        "fwd_num_warps": 4,
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 64,
        "bwd_BLOCK_M2": 64,
        "bwd_BLOCK_N2": 32,
        "bwd_num_stages": 3,
        "bwd_num_warps": 4,
        "ROWS_GUARANTEED_SAFE": True,
        "BLOCKS_ARE_CONTIGUOUS": True,
    },
    "a100_d256_contig_prescale": {
        "fwd_BLOCK_M": 32,
        "fwd_BLOCK_N": 64,
        "fwd_num_stages": 2,
        "fwd_num_warps": 4,
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 64,
        "bwd_BLOCK_M2": 64,
        "bwd_BLOCK_N2": 32,
        "bwd_num_stages": 3,
        "bwd_num_warps": 4,
        "PRESCALE_QK": True,
        "ROWS_GUARANTEED_SAFE": True,
        "BLOCKS_ARE_CONTIGUOUS": True,
    },
    "a100_d256_contig_write_dq_false": {
        "fwd_BLOCK_M": 32,
        "fwd_BLOCK_N": 64,
        "fwd_num_stages": 2,
        "fwd_num_warps": 4,
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 64,
        "bwd_BLOCK_M2": 64,
        "bwd_BLOCK_N2": 32,
        "bwd_num_stages": 3,
        "bwd_num_warps": 4,
        "ROWS_GUARANTEED_SAFE": True,
        "BLOCKS_ARE_CONTIGUOUS": True,
        "WRITE_DQ": False,
    },
}


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if torch.cuda.is_available():
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            fn()
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
        else:
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def peak_bytes(fn, device: str) -> int:
    if not device.startswith("cuda"):
        fn()
        return 0
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def best_nested_speedup(nested: dict) -> dict | None:
    best = None
    for block_key, by_preset in nested.items():
        for preset, speedup in by_preset.items():
            if not isinstance(speedup, float):
                continue
            if best is None or speedup > best["speedup"]:
                best = {"block_mask": block_key, "preset": preset, "speedup": speedup}
    return best


def make_inputs(args):
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    action_len = args.action_blocks * args.action_block_size
    total = args.prefix_len + action_len
    torch.manual_seed(args.seed)
    q = torch.randn(args.batch, args.heads, total, args.head_dim, device=args.device, dtype=dtype)
    kv_heads = args.kv_heads or args.heads
    k = torch.randn(args.batch, kv_heads, total, args.head_dim, device=args.device, dtype=dtype)
    v = torch.randn_like(k)
    prefix_valid = torch.ones(args.batch, args.prefix_len, device=args.device, dtype=torch.bool)
    prefix_att = torch.zeros_like(prefix_valid)
    prefix_att[:, args.prefix_len // 2 :] = True
    if args.no_prefix_mask:
        prefix_valid = None
        prefix_att = None
    return q, k, v, prefix_valid, prefix_att


def make_flex_bundle(args, prefix_valid, prefix_att, block_size: tuple[int, int]):
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except Exception:
        return None

    compiled_mask = torch.compile(create_block_mask, dynamic=False)
    batch = args.batch
    prefix_len = args.prefix_len
    action_len = args.action_blocks * args.action_block_size
    total_len = prefix_len + action_len
    chunk = args.action_block_size
    pad = prefix_valid
    cum = torch.cumsum(prefix_att.to(torch.long), dim=1) if prefix_att is not None else None

    def prefix_rows(b, h, q_idx, kv_idx):
        if cum is None:
            return kv_idx < prefix_len
        kv_p = kv_idx.clamp(max=prefix_len - 1)
        ok = (cum[b, kv_p] <= cum[b, q_idx]) & pad[b, kv_p] & pad[b, q_idx]
        return (kv_idx < prefix_len) & ok

    def action_rows(b, h, q_idx, kv_idx):
        if pad is None:
            to_prefix = kv_idx < prefix_len
        else:
            kv_p = kv_idx.clamp(max=prefix_len - 1)
            to_prefix = (kv_idx < prefix_len) & pad[b, kv_p]
        same_block = (q_idx // chunk) == ((kv_idx - prefix_len) // chunk)
        return to_prefix | ((kv_idx >= prefix_len) & same_block)

    block_prefix = compiled_mask(
        prefix_rows,
        B=batch,
        H=None,
        Q_LEN=prefix_len,
        KV_LEN=total_len,
        device=torch.device(args.device),
        BLOCK_SIZE=block_size,
    )
    block_action = compiled_mask(
        action_rows,
        B=batch,
        H=None,
        Q_LEN=action_len,
        KV_LEN=total_len,
        device=torch.device(args.device),
        BLOCK_SIZE=block_size,
    )
    compiled_calls = {}
    scale = args.head_dim**-0.5
    gqa = (args.kv_heads or args.heads) != args.heads
    for name, options in FLEX_TILE_PRESETS.items():
        def prefix_call(q, k, v, block_mask, options=options):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=gqa,
                kernel_options=options,
            )

        def action_call(q, k, v, block_mask, options=options):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=gqa,
                kernel_options=options,
            )

        compiled_calls[name] = (
            torch.compile(prefix_call, dynamic=False),
            torch.compile(action_call, dynamic=False),
        )
    return compiled_calls, block_prefix, block_action


def _manual_part(qs, ks, vs, m, scale):
    """Materialized-logits attention: cuBLAS GEMMs + fused masked softmax.

    Exact SDPA semantics (fp32 softmax); grouped-query handled as a strided
    batched GEMM over (kv_head, group*Sq) without materializing repeated K/V.
    """
    B, H, Sq, D = qs.shape
    Hk = ks.shape[1]
    if Hk != H:
        g = H // Hk
        q2 = qs.reshape(B, Hk, g * Sq, D)
        logits = (q2 @ ks.transpose(-1, -2)).reshape(B, H, Sq, -1)
    else:
        logits = qs @ ks.transpose(-1, -2)
    logits = logits * scale + m
    p = logits.float().softmax(dim=-1).to(qs.dtype)
    if Hk != H:
        out = (p.reshape(B, Hk, g * Sq, -1) @ vs).reshape(B, H, Sq, D)
    else:
        out = p @ vs
    return out


_manual_part_compiled = None


def get_manual_part():
    global _manual_part_compiled
    if _manual_part_compiled is None:
        _manual_part_compiled = torch.compile(_manual_part, dynamic=False)
    return _manual_part_compiled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=None, help="KV heads for GQA (default: same as --heads)")
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--prefix-len", type=int, default=700)
    parser.add_argument("--action-blocks", type=int, default=5)
    parser.add_argument("--action-block-size", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["fwd", "fwdbwd", "all"], default="all")
    parser.add_argument(
        "--backend",
        default="all",
        help="comma-separated subset of {package, torch-flex, manual} or 'all'",
    )
    parser.add_argument("--flex-preset", choices=sorted(FLEX_TILE_PRESETS), default="a100_d256_bwd_32x64")
    parser.add_argument("--sweep-flex-presets", action="store_true")
    parser.add_argument("--block-mask-q", type=int, default=128)
    parser.add_argument("--block-mask-kv", type=int, default=128)
    parser.add_argument("--sweep-block-mask-sizes", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--require-gates", action="store_true")
    parser.add_argument("--no-prefix-mask", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    backends = (
        {"package", "torch-flex", "manual"}
        if args.backend == "all"
        else {b.strip() for b in args.backend.split(",") if b.strip()}
    )

    q, k, v, prefix_valid, prefix_att = make_inputs(args)
    action_len = args.action_blocks * args.action_block_size
    scale = args.head_dim**-0.5
    pm, am = flex_ops.build_block_sparse_bool_masks(
        prefix_valid,
        prefix_att,
        batch=args.batch,
        prefix_len=args.prefix_len,
        action_len=action_len,
        action_block_size=args.action_block_size,
        device=q.device,
    )
    full = torch.cat([pm, am], dim=1)
    add_mask = torch.where(
        full[:, None],
        torch.zeros((), device=q.device, dtype=q.dtype),
        torch.full((), flex_ops.MASK_VALUE_F32, device=q.device, dtype=q.dtype),
    )
    block_sizes = [(args.block_mask_q, args.block_mask_kv)]
    if args.sweep_block_mask_sizes:
        block_sizes = [(q, kv) for q in (16, 32, 64, 128) for kv in (32, 64, 128)]
    flex_bundles = {
        f"{q}x{kv}": make_flex_bundle(args, prefix_valid, prefix_att, (q, kv))
        for q, kv in block_sizes
    } if "torch-flex" in backends else {}
    flex_presets = sorted(FLEX_TILE_PRESETS) if args.sweep_flex_presets else [args.flex_preset]

    gqa = (args.kv_heads or args.heads) != args.heads

    def sdpa_fwd():
        out_p = F.scaled_dot_product_attention(
            q[:, :, : args.prefix_len],
            k,
            v,
            attn_mask=add_mask[:, :, : args.prefix_len],
            scale=scale,
            enable_gqa=gqa,
        )
        kd = torch.cat([k[:, :, : args.prefix_len].detach(), k[:, :, args.prefix_len :]], dim=2)
        vd = torch.cat([v[:, :, : args.prefix_len].detach(), v[:, :, args.prefix_len :]], dim=2)
        out_a = F.scaled_dot_product_attention(
            q[:, :, args.prefix_len :],
            kd,
            vd,
            attn_mask=add_mask[:, :, args.prefix_len :],
            scale=scale,
            enable_gqa=gqa,
        )
        return torch.cat([out_p, out_a], dim=2)

    manual_part = get_manual_part() if "manual" in backends else None

    def manual_fwd():
        out_p = manual_part(q[:, :, : args.prefix_len], k, v, add_mask[:, :, : args.prefix_len], scale)
        kd = torch.cat([k[:, :, : args.prefix_len].detach(), k[:, :, args.prefix_len :]], dim=2)
        vd = torch.cat([v[:, :, : args.prefix_len].detach(), v[:, :, args.prefix_len :]], dim=2)
        out_a = manual_part(
            q[:, :, args.prefix_len :], kd, vd, add_mask[:, :, args.prefix_len :], scale
        )
        return torch.cat([out_p, out_a], dim=2)

    def package_fwd():
        return flex_ops.flex_attention(
            q,
            k,
            v,
            prefix_len=args.prefix_len,
            action_block_size=args.action_block_size,
            prefix_valid=prefix_valid,
            prefix_att=prefix_att,
            scale=scale,
        )

    def torch_flex_fwd(preset: str, block_key: str):
        if not flex_bundles:
            raise RuntimeError("PyTorch FlexAttention is unavailable")
        compiled_calls, block_prefix, block_action = flex_bundles[block_key]
        prefix_call, action_call = compiled_calls[preset]
        out_p = prefix_call(
            q[:, :, : args.prefix_len],
            k,
            v,
            block_prefix,
        )
        kd = torch.cat([k[:, :, : args.prefix_len].detach(), k[:, :, args.prefix_len :]], dim=2)
        vd = torch.cat([v[:, :, : args.prefix_len].detach(), v[:, :, args.prefix_len :]], dim=2)
        out_a = action_call(
            q[:, :, args.prefix_len :],
            kd,
            vd,
            block_action,
        )
        return torch.cat([out_p, out_a], dim=2)

    report = {
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
        "torch": torch.__version__,
        "shape": {
            "B": args.batch,
            "heads": args.heads,
            "kv_heads": args.kv_heads or args.heads,
            "head_dim": args.head_dim,
            "prefix_len": args.prefix_len,
            "action_len": action_len,
            "action_block_size": args.action_block_size,
        },
        "backend": args.backend,
        "flex_presets": flex_presets,
        "block_mask_sizes": list(flex_bundles) if flex_bundles else [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report["package_has_native_ops"] = False
    report["package_native_supported"] = False

    with torch.no_grad():
        sdpa_out = sdpa_fwd().float()
        if "package" in backends:
            report["package_fwd_max_abs_diff"] = float((sdpa_out - package_fwd().float()).abs().max())
        if "manual" in backends:
            report["manual_fwd_max_abs_diff"] = float((sdpa_out - manual_fwd().float()).abs().max())
        if flex_bundles:
            report["torch_flex_fwd_max_abs_diff"] = {}
            for block_key in flex_bundles:
                report["torch_flex_fwd_max_abs_diff"][block_key] = {}
            for block_key in flex_bundles:
              for preset in flex_presets:
                try:
                    report["torch_flex_fwd_max_abs_diff"][block_key][preset] = float(
                        (sdpa_out - torch_flex_fwd(preset, block_key).float()).abs().max()
                    )
                except Exception as exc:
                    report["torch_flex_fwd_max_abs_diff"][block_key][preset] = f"ERROR: {type(exc).__name__}: {exc}"
        if args.mode in {"fwd", "all"}:
            report["sdpa_fwd_ms"] = bench(sdpa_fwd, args.warmup, args.iters)
            report["sdpa_peak_bytes"] = peak_bytes(sdpa_fwd, args.device)
            if "package" in backends:
                report["package_fwd_ms"] = bench(package_fwd, args.warmup, args.iters)
                report["package_fwd_speedup"] = report["sdpa_fwd_ms"] / report["package_fwd_ms"]
                report["package_peak_bytes"] = peak_bytes(package_fwd, args.device)
            if "manual" in backends:
                report["manual_fwd_ms"] = bench(manual_fwd, args.warmup, args.iters)
                report["manual_fwd_speedup"] = report["sdpa_fwd_ms"] / report["manual_fwd_ms"]
                report["manual_peak_bytes"] = peak_bytes(manual_fwd, args.device)
            if flex_bundles:
                report["torch_flex_fwd_ms"] = {}
                report["torch_flex_fwd_speedup"] = {}
                for block_key in flex_bundles:
                    report["torch_flex_fwd_ms"][block_key] = {}
                    report["torch_flex_fwd_speedup"][block_key] = {}
                for block_key in flex_bundles:
                  for preset in flex_presets:
                    try:
                        ms = bench(lambda preset=preset, block_key=block_key: torch_flex_fwd(preset, block_key), args.warmup, args.iters)
                    except Exception as exc:
                        report["torch_flex_fwd_ms"][block_key][preset] = f"ERROR: {type(exc).__name__}: {exc}"
                    else:
                        report["torch_flex_fwd_ms"][block_key][preset] = ms
                        report["torch_flex_fwd_speedup"][block_key][preset] = report["sdpa_fwd_ms"] / ms
                report["best_torch_flex_fwd"] = best_nested_speedup(report["torch_flex_fwd_speedup"])

    if args.mode in {"fwdbwd", "all"}:
        def sdpa_fwdbwd():
            qq = q.detach().clone().requires_grad_(True)
            kk = k.detach().clone().requires_grad_(True)
            vv = v.detach().clone().requires_grad_(True)
            out_p = F.scaled_dot_product_attention(
                qq[:, :, : args.prefix_len],
                kk,
                vv,
                attn_mask=add_mask[:, :, : args.prefix_len],
                scale=scale,
                enable_gqa=gqa,
            )
            kd = torch.cat([kk[:, :, : args.prefix_len].detach(), kk[:, :, args.prefix_len :]], dim=2)
            vd = torch.cat([vv[:, :, : args.prefix_len].detach(), vv[:, :, args.prefix_len :]], dim=2)
            out_a = F.scaled_dot_product_attention(
                qq[:, :, args.prefix_len :],
                kd,
                vd,
                attn_mask=add_mask[:, :, args.prefix_len :],
                scale=scale,
                enable_gqa=gqa,
            )
            torch.cat([out_p, out_a], dim=2).float().square().mean().backward()

        def manual_fwdbwd():
            qq = q.detach().clone().requires_grad_(True)
            kk = k.detach().clone().requires_grad_(True)
            vv = v.detach().clone().requires_grad_(True)
            out_p = manual_part(
                qq[:, :, : args.prefix_len], kk, vv, add_mask[:, :, : args.prefix_len], scale
            )
            kd = torch.cat([kk[:, :, : args.prefix_len].detach(), kk[:, :, args.prefix_len :]], dim=2)
            vd = torch.cat([vv[:, :, : args.prefix_len].detach(), vv[:, :, args.prefix_len :]], dim=2)
            out_a = manual_part(
                qq[:, :, args.prefix_len :], kd, vd, add_mask[:, :, args.prefix_len :], scale
            )
            torch.cat([out_p, out_a], dim=2).float().square().mean().backward()

        def package_fwdbwd():
            qq = q.detach().clone().requires_grad_(True)
            kk = k.detach().clone().requires_grad_(True)
            vv = v.detach().clone().requires_grad_(True)
            flex_ops.flex_attention(
                qq,
                kk,
                vv,
                prefix_len=args.prefix_len,
                action_block_size=args.action_block_size,
                prefix_valid=prefix_valid,
                prefix_att=prefix_att,
                scale=scale,
            ).float().square().mean().backward()

        def torch_flex_fwdbwd(preset: str, block_key: str):
            if not flex_bundles:
                raise RuntimeError("PyTorch FlexAttention is unavailable")
            compiled_calls, block_prefix, block_action = flex_bundles[block_key]
            prefix_call, action_call = compiled_calls[preset]
            qq = q.detach().clone().requires_grad_(True)
            kk = k.detach().clone().requires_grad_(True)
            vv = v.detach().clone().requires_grad_(True)
            out_p = prefix_call(
                qq[:, :, : args.prefix_len],
                kk,
                vv,
                block_prefix,
            )
            kd = torch.cat([kk[:, :, : args.prefix_len].detach(), kk[:, :, args.prefix_len :]], dim=2)
            vd = torch.cat([vv[:, :, : args.prefix_len].detach(), vv[:, :, args.prefix_len :]], dim=2)
            out_a = action_call(
                qq[:, :, args.prefix_len :],
                kd,
                vd,
                block_action,
            )
            torch.cat([out_p, out_a], dim=2).float().square().mean().backward()

        report["sdpa_fwdbwd_ms"] = bench(sdpa_fwdbwd, args.warmup, args.iters)
        if "package" in backends:
            report["package_fwdbwd_ms"] = bench(package_fwdbwd, args.warmup, args.iters)
            report["package_fwdbwd_speedup"] = report["sdpa_fwdbwd_ms"] / report["package_fwdbwd_ms"]
        if "manual" in backends:
            report["manual_fwdbwd_ms"] = bench(manual_fwdbwd, args.warmup, args.iters)
            report["manual_fwdbwd_speedup"] = report["sdpa_fwdbwd_ms"] / report["manual_fwdbwd_ms"]
        if flex_bundles:
            report["torch_flex_fwdbwd_ms"] = {}
            report["torch_flex_fwdbwd_speedup"] = {}
            for block_key in flex_bundles:
                report["torch_flex_fwdbwd_ms"][block_key] = {}
                report["torch_flex_fwdbwd_speedup"][block_key] = {}
            for block_key in flex_bundles:
              for preset in flex_presets:
                try:
                    ms = bench(lambda preset=preset, block_key=block_key: torch_flex_fwdbwd(preset, block_key), args.warmup, args.iters)
                except Exception as exc:
                    report["torch_flex_fwdbwd_ms"][block_key][preset] = f"ERROR: {type(exc).__name__}: {exc}"
                else:
                    report["torch_flex_fwdbwd_ms"][block_key][preset] = ms
                    report["torch_flex_fwdbwd_speedup"][block_key][preset] = report["sdpa_fwdbwd_ms"] / ms
            report["best_torch_flex_fwdbwd"] = best_nested_speedup(report["torch_flex_fwdbwd_speedup"])

        if "package" not in backends:
            _finish(report, args)
            return
        q1 = q.detach().clone().requires_grad_(True)
        k1 = k.detach().clone().requires_grad_(True)
        v1 = v.detach().clone().requires_grad_(True)
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        out1_p = F.scaled_dot_product_attention(
            q1[:, :, : args.prefix_len], k1, v1, attn_mask=add_mask[:, :, : args.prefix_len], scale=scale
        )
        k1d = torch.cat([k1[:, :, : args.prefix_len].detach(), k1[:, :, args.prefix_len :]], dim=2)
        v1d = torch.cat([v1[:, :, : args.prefix_len].detach(), v1[:, :, args.prefix_len :]], dim=2)
        out1_a = F.scaled_dot_product_attention(
            q1[:, :, args.prefix_len :], k1d, v1d, attn_mask=add_mask[:, :, args.prefix_len :], scale=scale
        )
        torch.cat([out1_p, out1_a], dim=2).float().square().mean().backward()
        flex_ops.flex_attention(
            q2,
            k2,
            v2,
            prefix_len=args.prefix_len,
            action_block_size=args.action_block_size,
            prefix_valid=prefix_valid,
            prefix_att=prefix_att,
            scale=scale,
        ).float().square().mean().backward()
        denom = torch.linalg.vector_norm(torch.cat([q1.grad.flatten(), k1.grad.flatten(), v1.grad.flatten()])).clamp_min(1e-12)
        numer = torch.linalg.vector_norm(
            torch.cat([(q1.grad - q2.grad).flatten(), (k1.grad - k2.grad).flatten(), (v1.grad - v2.grad).flatten()])
        )
        report["package_grad_norm_rel_diff"] = float(numer / denom)

    _finish(report, args)


def _finish(report, args):
    gates = {
        "package_fwd_max_abs_diff": report.get("package_fwd_max_abs_diff", 0.0) <= 2e-3,
        "package_grad_norm_rel_diff": report.get("package_grad_norm_rel_diff", 0.0) <= 1e-2,
    }
    if "torch_flex_fwd_max_abs_diff" in report:
        vals = [v for by_block in report["torch_flex_fwd_max_abs_diff"].values() for v in by_block.values()]
        gates["torch_flex_fwd_max_abs_diff"] = any(isinstance(v, float) and v <= 2e-3 for v in vals)
    if "manual_fwd_max_abs_diff" in report:
        gates["manual_fwd_max_abs_diff"] = report["manual_fwd_max_abs_diff"] <= 2e-3
    if "manual_fwd_speedup" in report:
        gates["manual_fwd_speedup_ge_1p0526"] = report["manual_fwd_speedup"] >= (1.0 / 0.95)
    if "manual_fwdbwd_speedup" in report:
        gates["manual_fwdbwd_speedup_ge_1p0526"] = report["manual_fwdbwd_speedup"] >= (1.0 / 0.95)
    if "package_fwd_speedup" in report:
        gates["package_fwd_speedup_ge_1p0526"] = report["package_fwd_speedup"] >= (1.0 / 0.95)
        if report.get("sdpa_peak_bytes", 0) > 0:
            gates["package_peak_memory_le_sdpa_plus_3pct"] = report["package_peak_bytes"] <= int(report["sdpa_peak_bytes"] * 1.03)
    if "package_fwdbwd_speedup" in report:
        gates["package_fwdbwd_speedup_ge_1p0526"] = report["package_fwdbwd_speedup"] >= (1.0 / 0.95)
    if "torch_flex_fwd_speedup" in report:
        vals = [v for by_block in report["torch_flex_fwd_speedup"].values() for v in by_block.values() if isinstance(v, float)]
        gates["torch_flex_fwd_speedup_ge_1p0526"] = bool(vals) and max(vals) >= (1.0 / 0.95)
    if "torch_flex_fwdbwd_speedup" in report:
        vals = [v for by_block in report["torch_flex_fwdbwd_speedup"].values() for v in by_block.values() if isinstance(v, float)]
        gates["torch_flex_fwdbwd_speedup_ge_1p0526"] = bool(vals) and max(vals) >= (1.0 / 0.95)
    report["gates"] = gates

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.require_gates and not all(gates.values()):
        raise SystemExit("one or more Flex attention acceptance gates failed")


if __name__ == "__main__":
    main()
