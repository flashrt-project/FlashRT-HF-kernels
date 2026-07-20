#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _source_loader import load_source_ops

CODEBOOK = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
)


def ue(v):
    e = (v >> 3) & 15
    m = v & 7
    return torch.where(
        e == 0, m.float() * 2**-9, (1 + m.float() / 8) * torch.pow(2.0, (e - 7).float())
    )


def sf_bytes(rows, k):
    return ((rows + 127) // 128) * (((k // 16) + 3) // 4) * 512


def make_sf(rows, k, device):
    linear = torch.randint(32, 80, (rows, k // 16), device=device, dtype=torch.uint8)
    out = torch.zeros(sf_bytes(rows, k), device=device, dtype=torch.uint8)
    r = torch.arange(rows, device=device)[:, None]
    b = torch.arange(k // 16, device=device)[None, :]
    off = (
        (r // 128 * ((k // 16 + 3) // 4) + b // 4) * 512
        + (r % 32) * 16
        + (r % 128 // 32) * 4
        + b % 4
    )
    out[off.long()] = linear
    return out, linear


def deq(p, sf, alpha=1.0):
    lo = p.int() & 15
    hi = (p.int() >> 4) & 15
    vals = CODEBOOK.to(p.device)[torch.stack((lo, hi), -1).flatten(-2)]
    return vals * ue(sf.int()).repeat_interleave(16, 1) * alpha


def run(ops):
    torch.manual_seed(19)
    dev = "cuda"
    checks = 0
    cases = [
        (16, 8, 64, 3, 4),
        (16, 72, 128, 3, 4),
        (16, 1024, 2048, 2, 3),
        (64, 16, 64, 2, 3),
        (64, 80, 256, 2, 3),
        (64, 64, 64, 2, 3),
        (64, 128, 256, 2, 3),
        (64, 1024, 2048, 1, 2),
    ]
    for tr, n, k, tiles, experts in cases:
        rows = tr * tiles
        ap = torch.randint(0, 256, (rows, k // 2), device=dev, dtype=torch.uint8)
        wp = torch.randint(0, 256, (experts, n, k // 2), device=dev, dtype=torch.uint8)
        asw, al = make_sf(rows, k, dev)
        wsw = []
        wl = []
        for _ in range(experts):
            x, y = make_sf(n, k, dev)
            wsw.append(x)
            wl.append(y)
        wsw = torch.stack(wsw)
        alpha = torch.rand(experts, device=dev) + 0.5
        te = torch.arange(tiles, device=dev, dtype=torch.int32) % experts
        got = ops.grouped_nvfp4_gemm_bf16(ap, wp, asw, wsw, alpha, te, tile_rows=tr)
        adeq = deq(ap, al)
        ref = torch.empty(rows, n, device=dev)
        for t in range(tiles):
            e = int(te[t])
            ref[t * tr : (t + 1) * tr] = (
                adeq[t * tr : (t + 1) * tr] @ deq(wp[e], wl[e], alpha[e]).T
            )
        ref = ref.bfloat16()
        torch.cuda.synchronize()
        d = (got.float() - ref.float()).abs().flatten()
        cos = torch.nn.functional.cosine_similarity(
            got.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        rel_l2 = (d.norm() / ref.float().norm().clamp_min(1e-12)).item()
        max_rel = (d.max() / ref.float().abs().max().clamp_min(1e-12)).item()
        print(
            f"tile={tr} N={n} K={k} max={d.max().item():.6f} p99={torch.quantile(d,.99).item():.6f} mean={d.mean().item():.6f} cos={cos:.8f} rel_l2={rel_l2:.8f} max_rel={max_rel:.8f}"
        )
        assert (
            got.dtype == torch.bfloat16
            and cos >= 0.999
            and rel_l2 <= 0.0025
            and max_rel <= 0.01
        )
        checks += 1
        if tr == 64 and n == 128 and k == 256:
            compiled = torch.compile(
                lambda aa: ops.grouped_nvfp4_gemm_bf16(
                    aa, wp, asw, wsw, alpha, te, tile_rows=tr
                ),
                fullgraph=True,
            )
            cg = compiled(ap)
            torch.cuda.synchronize()
            assert torch.equal(cg, got)
            checks += 1
            gout = torch.empty_like(got)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ops.grouped_nvfp4_gemm_bf16(
                    ap, wp, asw, wsw, alpha, te, tile_rows=tr, out=gout
                )
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(gout, got)
            checks += 1
    print(f"PASS checks={checks}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--registration-include")
    a = p.parse_args()
    run(load_source_ops(a.registration_include))
