from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import sys
import torch
import types

PACKAGE = Path(__file__).resolve().parents[1]


def load_source_ops(registration_include=None):
    from torch.utils.cpp_extension import load

    include = registration_include or os.environ.get(
        "KERNEL_BUILDER_REGISTRATION_INCLUDE"
    )
    if not include:
        include = str(
            PACKAGE.parent.parent
            / "kernels/kernel-builder/src/pyproject/templates/torch"
        )
    cutlass = os.environ.get(
        "CUTLASS_INCLUDE",
        "/home/heima/suliang/PI/official/FlashRT/third_party/cutlass/include",
    )
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    ns = "grouped_moe_gemm_source_test"
    load(
        name=ns,
        sources=[
            str(PACKAGE / "torch-ext/torch_binding.cpp"),
            *map(str, sorted((PACKAGE / "csrc").glob("*.cu"))),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), include, cutlass],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "-DCUDA_KERNEL"],
        is_python_module=False,
        verbose=False,
    )
    # Exercise the shipped Python wrapper, including its custom-op fake
    # registration, against the source-built low-level namespace. This keeps
    # source tests from silently validating a test-only wrapper that differs
    # from the Hub artifact.
    package_name = "grouped_moe_gemm_source_public"
    ops_module = types.ModuleType(f"{package_name}._ops")
    ops_module.ops = getattr(torch.ops, ns)
    ops_module.add_op_namespace_prefix = (
        lambda name: f"{package_name}::{name}"
    )
    sys.modules[ops_module.__name__] = ops_module

    package_dir = PACKAGE / "torch-ext/grouped_moe_gemm"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load packaged grouped MoE wrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module
