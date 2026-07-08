#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, sys, torch, torch.nn.functional as F

def load_ops(artifact=None):
    if artifact: sys.path.insert(0, artifact)
    try: return importlib.import_module("flashrt_siglip_fwd_fusion")
    finally:
        if artifact: sys.path.remove(artifact)

def run(ops, mode):
    torch.manual_seed(19); count=0
    shapes=[(4,16,32),(2,256,128)] if mode=="full" else [(4,16,32)]
    for b,t,h in shapes:
        x=torch.randn(b,t,h,device="cuda",dtype=torch.float32); r=torch.randn_like(x); w=torch.randn(h,device="cuda"); bias=torch.randn(h,device="cuda")
        with torch.no_grad():
            torch.testing.assert_close(ops.siglip_residual_layernorm_fwd(x,r,w,bias), F.layer_norm(x+r,(h,),w,bias,1e-6)); torch.testing.assert_close(ops.siglip_gelu_fwd(x,bias), F.gelu(x+bias,approximate="tanh")); assert ops.use_fused_siglip_path()
        assert not ops.use_fused_siglip_path()
        torch.compile(lambda a,b1,c,d: ops.siglip_residual_layernorm_fwd(a,b1,c,d), fullgraph=False)(x,r,w,bias); count+=1
    print(f"flashrt-siglip-fwd-fusion {mode}: passed {count}/{count}")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--backend",choices=["installed"],default="installed"); p.add_argument("--artifact"); p.add_argument("--mode",choices=["smoke","full"],default="smoke")
    a=p.parse_args(); run(load_ops(a.artifact),a.mode)
