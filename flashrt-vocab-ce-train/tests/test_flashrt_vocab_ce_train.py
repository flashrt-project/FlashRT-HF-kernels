#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, importlib.util, os, sys, types
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-vocab-ce-train"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


def load_ops(backend, artifact=None):
    if backend == "installed":
        if artifact:
            sys.path.insert(0, artifact)
        try:
            return importlib.import_module("flashrt_vocab_ce_train")
        finally:
            if artifact:
                sys.path.remove(artifact)
    return load_source_ops()


def load_source_ops():
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(
            f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}"
        )
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}a")
    namespace = "flashrt_vocab_ce_train_source_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "vocab_ce_train.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        verbose=False,
    )
    # Exercise the shipped Python wrapper, including its custom-op fake
    # registration, against the source-built low-level namespace.
    package_name = "flashrt_vocab_ce_train_source"
    ops_module = types.ModuleType(f"{package_name}._ops")
    ops_module.ops = getattr(torch.ops, namespace)
    ops_module.add_op_namespace_prefix = lambda name: f"{namespace}::{name}"
    sys.modules[ops_module.__name__] = ops_module

    package_dir = PACKAGE / "torch-ext" / "flashrt_vocab_ce_train"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load packaged flashrt-vocab-ce-train wrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def run(ops, mode):
    torch.manual_seed(11); count=0
    shapes=[(7,16,31),(23,32,257)] if mode=="full" else [(7,16,31)]
    for n,h,v in shapes:
        x=torch.randn(n,h,device="cuda",dtype=torch.float64,requires_grad=True); w=torch.randn(v,h,device="cuda",dtype=torch.float64,requires_grad=True); labels=torch.randint(0,v,(n,),device="cuda"); labels[0]=-100
        torch.autograd.gradcheck(lambda a,b: ops.vocab_ce(a,b,labels,0.01), (x,w), eps=1e-6, atol=1e-4, rtol=1e-3)
        got=ops.vocab_ce(x.float(),w.float(),labels,0.01); logits=x.float()@w.float().t(); valid=labels!=-100; nv=valid.sum().clamp(min=1)
        ref=F.cross_entropy(logits,labels,ignore_index=-100,reduction="sum")/nv + 0.01*(torch.logsumexp(logits,-1).square()*valid).sum()/nv
        torch.testing.assert_close(got,ref)
        with torch.autograd.set_detect_anomaly(True): checkpoint(lambda a,b: ops.vocab_ce(a,b,labels,0.0), x.float(),w.float(), use_reentrant=False).backward()
        torch.compile(lambda a,b: ops.vocab_ce(a,b,labels,0.0), fullgraph=False)(x.float().detach(),w.float().detach()); count+=1
    print(f"flashrt-vocab-ce-train {mode}: passed {count}/{count}")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--backend",choices=["source","installed"],default="installed"); p.add_argument("--artifact"); p.add_argument("--mode",choices=["smoke","full"],default="smoke")
    a=p.parse_args(); run(load_ops(a.backend, a.artifact), a.mode)
