#!/usr/bin/env python3
"""Correctness tests for int4-blackwell native INT4 tensor-core primitives."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "int4-blackwell"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)
CUTLASS_INCLUDE = Path(os.environ.get("INT4_BLACKWELL_CUTLASS_INCLUDE", ""))

SUPPORTED = {(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major == 12 and minor == 1:
        return "12.1"
    if major >= 12:
        return "12.0a"
    if (major, minor) == (11, 0):
        return "11.0a"
    if (major, minor) == (10, 0):
        return "10.0a"
    if (major, minor) == (10, 3):
        return "10.3a"
    return f"{major}.{minor}"


def _pack(values: torch.Tensor) -> torch.Tensor:
    codes = torch.where(values >= 0, values, -values + 8).to(torch.uint8)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def tcgen05_int4_gemm(self, a_packed, sfa, b_packed, sfb):
        return self._ops.tcgen05_int4_gemm_bf16(a_packed, sfa, b_packed, sfb)

    def codebook_probe(self, mode: str = "ab"):
        if mode != "ab":
            raise ValueError("source backend exposes only the tcgen05 INT4 x INT4 descriptor")
        m = n = k = 128
        b_packed = torch.full((n, k // 2), 0x11, device="cuda", dtype=torch.uint8)
        sfa = torch.full((m * k,), 0x38, device="cuda", dtype=torch.uint8)
        sfb = torch.full((n * k,), 0x38, device="cuda", dtype=torch.uint8)
        values = []
        for value in range(16):
            packed = value | (value << 4)
            a_packed = torch.full((m, k // 2), packed, device="cuda", dtype=torch.uint8)
            tile = self._ops.tcgen05_int4_gemm_bf16(a_packed, sfa, b_packed, sfb)
            first = tile[0, 0]
            if not torch.equal(tile, first.expand_as(tile)):
                raise RuntimeError("tcgen05 INT4 codebook output is not uniform")
            values.append(first.float() / k)
        return torch.stack(values).cpu()


class InstalledOps:
    def __init__(self, module) -> None:
        self._module = module

    def tcgen05_int4_gemm(self, a_packed, sfa, b_packed, sfb):
        return self._module.tcgen05_int4_gemm_bf16(a_packed, sfa, b_packed, sfb)

    def codebook_probe(self, mode: str = "ab"):
        return self._module.codebook_probe(mode)


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    if not CUTLASS_INCLUDE.is_dir():
        raise RuntimeError(
            f"set INT4_BLACKWELL_CUTLASS_INCLUDE to a CUTLASS include directory "
            f"(got {CUTLASS_INCLUDE!r})"
        )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "int4_blackwell_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "arch_guard.cu"),
            str(PACKAGE / "csrc" / "gemm" / "int4_tcgen05_gemm.cu"),
        ],
        extra_include_paths=[
            str(PACKAGE / "csrc"),
            str(CUTLASS_INCLUDE),
            str(REGISTRATION_INCLUDE),
        ],
        extra_cflags=["-O3", "-DCUDA_KERNEL", "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-DCUDA_KERNEL",
            "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
        ],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return InstalledOps(importlib.import_module("int4_blackwell"))
    finally:
        if artifact:
            sys.path.remove(artifact)


def _check_codebook(ops, mode: str, expected: list[float]) -> None:
    got = ops.codebook_probe(mode)
    torch.testing.assert_close(got, torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0)
    print(f"PASS codebook_probe mode={mode}")


def _check_tcgen05_gemm(ops, shape) -> None:
    torch.manual_seed(20260713)
    m, n, k = shape
    a = torch.randint(-2, 3, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-2, 3, (n, k), device="cuda", dtype=torch.int8)
    sfa = torch.full((m * k,), 0x38, device="cuda", dtype=torch.uint8)
    sfb = torch.full((n * k,), 0x38, device="cuda", dtype=torch.uint8)
    actual = ops.tcgen05_int4_gemm(_pack(a), sfa, _pack(b), sfb)
    expected = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"PASS tcgen05_int4_gemm shape={shape}: exact")


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in SUPPORTED:
        raise RuntimeError(f"int4-blackwell requires SM100/SM103/SM110/SM120/SM121; got {capability}")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)

    if capability in {(10, 0), (10, 3), (11, 0)}:
        # tcgen05 path validates the native INT4 x INT4 mode (see CARD.md);
        # mode "ab" decodes as E0M3/INT4, matching the SM12x "ab" expectation.
        _check_codebook(ops, "ab", [0, 1, 2, 3, 4, 5, 6, 7, 0, -1, -2, -3, -4, -5, -6, -7])
        shapes = [(128, 128, 128)] if args.mode == "smoke" else [(128, 128, 128), (128, 256, 256), (256, 128, 128)]
        for shape in shapes:
            _check_tcgen05_gemm(ops, shape)
    else:
        if args.backend != "source":
            for mode, expected in [
                ("e2m1", [0, .25, .5, .75, 1, 1.5, 2, 3, 0, -.25, -.5, -.75, -1, -1.5, -2, -3]),
                ("a", [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 0, -.5, -1, -1.5, -2, -2.5, -3, -3.5]),
                ("b", [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]),
                ("ab", [0, 1, 2, 3, 4, 5, 6, 7, 0, -1, -2, -3, -4, -5, -6, -7]),
            ]:
                _check_codebook(ops, mode, expected)
            scratch = torch.empty((1, 256), device="cuda", dtype=torch.float32)
            output = ops._module.mma_probe(iterations=16, blocks=1, out=scratch)
            if output is not scratch or output.shape != (1, 256):
                raise AssertionError("mma_probe must write in place into out")
            torch.cuda.synchronize()
            print("PASS mma_probe launches")
        else:
            raise RuntimeError("source backend covers only the tcgen05 path")
    print(f"PASS int4-blackwell {args.backend} mode={args.mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"passed": 1, "total": 1, "backend": args.backend}) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
