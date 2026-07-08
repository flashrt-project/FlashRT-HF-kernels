#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, sys, torch
from torch.utils.checkpoint import checkpoint

def load_ops(artifact=None):
    if artifact: sys.path.insert(0, artifact)
    try: return importlib.import_module("flashrt_qkv_epilogue_train")
    finally:
        if artifact: sys.path.remove(artifact)

def run(ops, mode):
    torch.manual_seed(17); count=0
    shapes=[(1,2,4,1,1,4),(2,11,16,4,2,8),(2,37,32,8,2,8)] if mode=="full" else [(1,2,4,1,1,4)]
    for b,t,hid,qh,kh,d in shapes:
        x=torch.randn(b,t,hid,device="cuda",dtype=torch.float64,requires_grad=True); wq=torch.randn(qh*d,hid,device="cuda",dtype=torch.float64,requires_grad=True); wk=torch.randn(kh*d,hid,device="cuda",dtype=torch.float64,requires_grad=True); wv=torch.randn(kh*d,hid,device="cuda",dtype=torch.float64,requires_grad=True); cos=torch.randn(b,t,d,device="cuda",dtype=torch.float64); sin=torch.randn(b,t,d,device="cuda",dtype=torch.float64)
        if x.numel() <= 64:
            torch.autograd.gradcheck(lambda a,b1,c1,d1: ops.qkv_rope_reference(a,b1,c1,d1,cos,sin,qh,kh,d)[0], (x,wq,wk,wv), eps=1e-6, atol=1e-4, rtol=1e-3)
        xb=x.to(torch.bfloat16).detach().requires_grad_(True); wqb=wq.to(torch.bfloat16).detach().requires_grad_(True); wkb=wk.to(torch.bfloat16).detach().requires_grad_(True); wvb=wv.to(torch.bfloat16).detach().requires_grad_(True); cb=cos.to(torch.bfloat16); sb=sin.to(torch.bfloat16)
        with torch.autograd.set_detect_anomaly(True):
            q,k,v=ops.qkv_rope_reference(xb,wqb,wkb,wvb,cb,sb,qh,kh,d); checkpoint(lambda a,b1,c1,d1: ops.qkv_rope_reference(a,b1,c1,d1,cb,sb,qh,kh,d)[0], xb,wqb,wkb,wvb, use_reentrant=False).float().sum().backward(retain_graph=True); (k.float().sum()+v.float().sum()).backward()
        torch.compile(lambda a,b1,c1,d1: ops.qkv_rope_reference(a,b1,c1,d1,cb,sb,qh,kh,d)[0], fullgraph=False)(xb.detach(),wqb.detach(),wkb.detach(),wvb.detach()); count+=1
    print(f"flashrt-qkv-epilogue-train {mode}: passed {count}/{count}")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--backend",choices=["installed"],default="installed"); p.add_argument("--artifact"); p.add_argument("--mode",choices=["smoke","full"],default="smoke")
    a=p.parse_args(); run(load_ops(a.artifact),a.mode)
