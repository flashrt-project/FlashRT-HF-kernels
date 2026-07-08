#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, sys, torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def load_ops(artifact=None):
    if artifact: sys.path.insert(0, artifact)
    try: return importlib.import_module("flashrt_vocab_ce_train")
    finally:
        if artifact: sys.path.remove(artifact)

def run(ops, mode):
    torch.manual_seed(11); count=0
    shapes=[(7,16,31),(23,32,257)] if mode=="full" else [(7,16,31)]
    for n,h,v in shapes:
        x=torch.randn(n,h,device="cuda",dtype=torch.float64,requires_grad=True); w=torch.randn(v,h,device="cuda",dtype=torch.float64,requires_grad=True); labels=torch.randint(0,v,(n,),device="cuda"); labels[0]=-100
        torch.autograd.gradcheck(lambda a,b: ops.vocab_ce_loss(a,b,labels,0.01), (x,w), eps=1e-6, atol=1e-4, rtol=1e-3)
        got=ops.vocab_ce_loss(x.float(),w.float(),labels,0.01); logits=x.float()@w.float().t(); valid=labels!=-100; nv=valid.sum().clamp(min=1)
        ref=F.cross_entropy(logits,labels,ignore_index=-100,reduction="sum")/nv + 0.01*(torch.logsumexp(logits,-1).square()*valid).sum()/nv
        torch.testing.assert_close(got,ref)
        with torch.autograd.set_detect_anomaly(True): checkpoint(lambda a,b: ops.vocab_ce_loss(a,b,labels,0.0), x.float(),w.float(), use_reentrant=False).backward()
        torch.compile(lambda a,b: ops.vocab_ce_loss(a,b,labels,0.0), fullgraph=False)(x.float().detach(),w.float().detach()); count+=1
    print(f"flashrt-vocab-ce-train {mode}: passed {count}/{count}")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--backend",choices=["installed"],default="installed"); p.add_argument("--artifact"); p.add_argument("--mode",choices=["smoke","full"],default="smoke")
    a=p.parse_args(); run(load_ops(a.artifact),a.mode)
