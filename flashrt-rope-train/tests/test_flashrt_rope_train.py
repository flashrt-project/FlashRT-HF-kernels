#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, sys, torch
from torch.utils.checkpoint import checkpoint

def load_ops(artifact=None):
    if artifact: sys.path.insert(0, artifact)
    try: return importlib.import_module("flashrt_rope_train")
    finally:
        if artifact: sys.path.remove(artifact)

def run(ops, mode):
    torch.manual_seed(13); count=0
    shapes=[(1,1,2,4),(2,4,17,32),(1,8,128,64)] if mode=="full" else [(1,1,2,4)]
    for b,h,t,d in shapes:
        q=torch.randn(b,h,t,d,device="cuda",dtype=torch.float64,requires_grad=True); k=torch.randn(b,1,t,d,device="cuda",dtype=torch.float64,requires_grad=True); cos=torch.randn(b,t,d,device="cuda",dtype=torch.float64); sin=torch.randn(b,t,d,device="cuda",dtype=torch.float64)
        if q.numel() <= 64:
            torch.autograd.gradcheck(lambda a,bb: ops.apply_rope_train(a,bb,cos,sin,1)[0], (q,k), eps=1e-6, atol=1e-4, rtol=1e-3)
        qb=q.to(torch.bfloat16).detach().requires_grad_(True); kb=k.to(torch.bfloat16).detach().requires_grad_(True); cb=cos.to(torch.bfloat16); sb=sin.to(torch.bfloat16)
        with torch.autograd.set_detect_anomaly(True):
            qo,ko=ops.apply_rope_train(qb,kb,cb,sb,1); checkpoint(lambda a,bb: ops.apply_rope_train(a,bb,cb,sb,1)[0], qb,kb, use_reentrant=False).float().sum().backward(retain_graph=True); (ko.float().sum()).backward()
        torch.compile(lambda a,bb: ops.apply_rope_train(a,bb,cb,sb,1)[0], fullgraph=False)(qb.detach(),kb.detach()); count+=1
    print(f"flashrt-rope-train {mode}: passed {count}/{count}")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--backend",choices=["installed"],default="installed"); p.add_argument("--artifact"); p.add_argument("--mode",choices=["smoke","full"],default="smoke")
    a=p.parse_args(); run(load_ops(a.artifact),a.mode)
