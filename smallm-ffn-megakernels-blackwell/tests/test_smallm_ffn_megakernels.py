#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _source_loader import load_installed_ops, load_source_ops

F8 = torch.float8_e4m3fn


def stat(got, ref, label):
    d = (got.float() - ref.float()).abs().flatten()
    c = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    print(
        f"{label} max={d.max().item():.7f} p99={torch.quantile(d,.99).item():.7f} mean={d.mean().item():.7f} cos={c:.8f}"
    )
    max_err = d.max().item()
    p99 = torch.quantile(d, .99).item()
    mean_err = d.mean().item()
    assert got.dtype == torch.bfloat16
    if torch.cuda.get_device_capability() == (11, 0) and "M=1" in label:
        assert max_err <= 0.0625 and p99 <= 0.046875
        assert mean_err <= 0.012 and c >= 0.9998
    else:
        assert max_err <= 0.0390625 and p99 <= 0.0078125
        assert mean_err <= 0.001 and c >= 0.999


def f8(x):
    return x.clamp(-448, 448).to(F8)


def run(o, full):
    torch.manual_seed(23)
    dev = "cuda"
    checks = 0
    for m in [1, 8, 21, 32] if full else [8]:
        x = f8(torch.randn(m, 1024, device=dev) * 0.2)
        uw = f8(torch.randn(4096, 1024, device=dev) * 0.02)
        dw = f8(torch.randn(1024, 4096, device=dev) * 0.02)
        ub = (torch.randn(4096, device=dev) * 0.01).bfloat16()
        db = (torch.randn(1024, device=dev) * 0.01).bfloat16()
        dinv = torch.ones(4096, device=dev, dtype=torch.bfloat16)
        g = torch.randn(m, 1024, device=dev, dtype=torch.bfloat16)
        r = torch.randn(m, 1024, device=dev, dtype=torch.bfloat16)
        out = torch.empty_like(r)
        scr = torch.empty(m, 4096, device=dev, dtype=F8)
        got = o.gated(x, uw, ub, dinv, dw, db, g, r, 1.0, 1.0, 1.0, out, scr)
        if torch.cuda.get_device_capability() == (11, 0):
            first = got.clone()
            first_scr = scr.clone()
            repeated = o.gated(
                x, uw, ub, dinv, dw, db, g, r,
                1.0, 1.0, 1.0, out, scr
            )
            torch.cuda.synchronize()
            assert torch.equal(first_scr, scr), (
                f"gated M={m} intermediate is not deterministic on SM110"
            )
            assert torch.equal(first, repeated), (
                f"gated M={m} output is not deterministic on SM110"
            )
        h = f8(
            torch.nn.functional.gelu(
                x.float() @ uw.float().T + ub.float(), approximate="tanh"
            )
        )
        ref = (
            (h.float() @ dw.float().T + db.float()) * g.float() + r.float()
        ).bfloat16()
        torch.cuda.synchronize()
        stat(got, ref, f"gated M={m}")
        checks += 1
        if full and m == 8:
            compiled = torch.compile(
                lambda a: o.gated_functional(
                    a, uw, ub, dinv, dw, db, g, r, 1.0, 1.0, 1.0
                ),
                fullgraph=True,
            )
            cg = compiled(x)
            torch.cuda.synchronize()
            stat(cg, ref, "gated compile")
            checks += 1
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                o.gated(x, uw, ub, dinv, dw, db, g, r, 1.0, 1.0, 1.0, out, scr)
            graph.replay()
            torch.cuda.synchronize()
            stat(out, ref, "gated graph")
            checks += 1
    for m, split in (
        [(1, False), (51, False), (144, False), (188, True)] if full else [(51, False)]
    ):
        if torch.cuda.get_device_capability() == (11, 0):
            split = False
        x = torch.randn(m, 512, device=dev, dtype=torch.bfloat16) * 0.2
        uw = f8(torch.randn(2048, 512, device=dev) * 0.02)
        dw = f8(torch.randn(512, 2048, device=dev) * 0.02)
        ub = (torch.randn(2048, device=dev) * 0.01).bfloat16()
        db = (torch.randn(512, device=dev) * 0.01).bfloat16()
        uinv = torch.ones(512, device=dev, dtype=torch.bfloat16)
        dinv = torch.ones(2048, device=dev, dtype=torch.bfloat16)
        r = torch.randn(m, 512, device=dev, dtype=torch.bfloat16)
        out = torch.empty_like(r)
        xs = torch.empty(m, 512, device=dev, dtype=F8)
        hs = torch.empty(m, 2048, device=dev, dtype=F8)
        b = torch.zeros(2, device=dev, dtype=torch.uint32)
        got = o.residual(
            x, uinv, uw, ub, dinv, dw, db, r, 1.0, 1.0, 1.0, 1.0, split, out, xs, hs, b
        )
        if torch.cuda.get_device_capability() == (11, 0):
            first = got.clone()
            first_xs = xs.clone()
            first_hs = hs.clone()
            repeated = o.residual(
                x, uinv, uw, ub, dinv, dw, db, r,
                1.0, 1.0, 1.0, 1.0, split, out, xs, hs, b
            )
            torch.cuda.synchronize()
            assert torch.equal(first_xs, xs), f"residual M={m} input quant is not deterministic on SM110"
            assert torch.equal(first_hs, hs), f"residual M={m} up projection is not deterministic on SM110"
            assert torch.equal(first, repeated), f"residual M={m} down projection is not deterministic on SM110"
        qx = f8(x.float())
        qh = f8(
            torch.nn.functional.gelu(
                qx.float() @ uw.float().T + ub.float(), approximate="tanh"
            )
        )
        ref = (qh.float() @ dw.float().T + db.float() + r.float()).bfloat16()
        torch.cuda.synchronize()
        stat(got, ref, f"residual M={m} split={split}")
        checks += 1
    print(f"PASS checks={checks}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("source", "installed"), default="source")
    p.add_argument("--artifact")
    p.add_argument("--registration-include")
    p.add_argument("--mode", choices=("smoke", "full"), default="full")
    a = p.parse_args()
    if a.backend == "installed":
        if not a.artifact:
            p.error("--artifact is required for --backend installed")
        ops = load_installed_ops(a.artifact)
    else:
        ops = load_source_ops(a.registration_include)
    run(ops, a.mode == "full")
