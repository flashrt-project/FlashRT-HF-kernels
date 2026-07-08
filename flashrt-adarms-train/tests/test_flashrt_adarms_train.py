#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, sys, torch
from torch.utils.checkpoint import checkpoint

def load_ops(artifact=None):
    if artifact: sys.path.insert(0, artifact)
    try: return importlib.import_module("flashrt_adarms_train")
    finally:
        if artifact: sys.path.remove(artifact)

def run(ops, mode):
    torch.manual_seed(7); count=0
    shapes=[(2,3,8),(1,5,16)] if mode=="full" else [(2,3,8)]
    for b,t,h in shapes:
        x=torch.randn(b,t,h,device="cuda",dtype=torch.float64,requires_grad=True)
        mod=torch.randn(b,1,3*h,device="cuda",dtype=torch.float64,requires_grad=True)
        torch.autograd.gradcheck(lambda a,m: ops.adarms_forward(a,m,1e-6,True)[0], (x,mod), eps=1e-6, atol=1e-4, rtol=1e-3)
        y,gate,rstd=ops.adarms_forward(x,mod)
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        xf = x.to(compute_dtype)
        ref_norm=xf*torch.rsqrt((xf*xf).mean(-1,keepdim=True)+1e-6)
        scale,shift,gate_ref=mod.chunk(3,dim=-1)
        torch.testing.assert_close(y,(ref_norm*(1+scale.to(compute_dtype))+shift.to(compute_dtype)).to(x.dtype))
        torch.testing.assert_close(gate,gate_ref.to(x.dtype))
        (y.float().square().mean()+gate.float().square().mean()+rstd.float().mean()).backward(); count+=1
        xb=torch.randn(b,t,h,device="cuda",dtype=torch.bfloat16,requires_grad=True); hb=torch.randn_like(xb,requires_grad=True); gb=torch.randn_like(xb,requires_grad=True); mb=torch.randn(b,3*h,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        def fn(a,bb,cc,dd): return ops.resgate_adarms_forward(a,bb,cc,dd)[1]
        with torch.autograd.set_detect_anomaly(True): checkpoint(fn, xb,hb,gb,mb, use_reentrant=False).float().sum().backward()
        torch.compile(lambda a,bb,cc,dd: ops.resgate_adarms_forward(a,bb,cc,dd)[1], fullgraph=False)(xb.detach(),hb.detach(),gb.detach(),mb.detach()); count+=1
    print(f"flashrt-adarms-train {mode}: passed {count}/{count}")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--backend",choices=["installed"],default="installed"); p.add_argument("--artifact"); p.add_argument("--mode",choices=["smoke","full"],default="smoke")
    a=p.parse_args(); run(load_ops(a.artifact), a.mode)
