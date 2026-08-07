#!/usr/bin/env python3
"""Correctness tests for flashrt-nvfp4 (NVFP4 scale-factor layout reshape).

Runs the same checks against the source build and the installed Hub artifact.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-nvfp4"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 12:
        return "12.0a"
    if (major, minor) == (11, 0):
        return "11.0a"
    return f"{major}.{minor}"


def _swizzled_bytes(rows: int, D: int) -> int:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if D <= 0 or D % 16 != 0:
        raise ValueError("D must be positive and divisible by 16")
    n_blocks = D // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 512


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def swizzled_bytes(self, rows: int, D: int) -> int:
        return _swizzled_bytes(rows, D)

    def linear_to_swizzled(self, scales, *, out=None, is_sfb=False):
        if out is None:
            out = torch.zeros(
                (_swizzled_bytes(scales.shape[0], scales.shape[1] * 16),),
                device=scales.device,
                dtype=torch.uint8,
            )
        self._ops.nvfp4_sf_linear_to_swizzled(scales, out, scales.shape[1] * 16, bool(is_sfb))
        return out


class InstalledOps:
    def __init__(self, module) -> None:
        self._module = module

    def swizzled_bytes(self, rows: int, D: int) -> int:
        return int(self._module.nvfp4_sf_swizzled_bytes(rows, D))

    def linear_to_swizzled(self, scales, *, out=None, is_sfb=False):
        return self._module.nvfp4_sf_linear_to_swizzled(scales, out=out, is_sfb=is_sfb)


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    namespace = "flashrt_nvfp4_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "nvfp4_sf_reshape_sm120.cu"),
        ],
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=["-O3", "-DCUDA_KERNEL"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"],
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return InstalledOps(importlib.import_module("flashrt_nvfp4"))
    finally:
        if artifact:
            sys.path.remove(artifact)


def _reference_swizzle(scales: torch.Tensor) -> torch.Tensor:
    rows, n_blocks = scales.shape
    n_col_super = (n_blocks + 3) // 4
    out = torch.zeros(_swizzled_bytes(rows, n_blocks * 16), dtype=torch.uint8)
    src = scales.cpu()
    for row in range(rows):
        rb = row // 128
        ri = row % 128
        for blk in range(n_blocks):
            cb = blk // 4
            ci = blk % 4
            super_idx = rb * n_col_super + cb
            inner_off = (ri % 32) * 16 + (ri // 32) * 4 + ci
            out[super_idx * 512 + inner_off] = src[row, blk]
    return out


def _check_bytes(ops) -> None:
    cases = [
        (1, 1024), (2, 4096), (31, 4096), (32, 4096), (33, 4096),
        (127, 4096), (128, 4096), (129, 4096), (16, 12288), (64, 16384),
    ]
    for rows, D in cases:
        got = ops.swizzled_bytes(rows, D)
        expected = _swizzled_bytes(rows, D)
        if got != expected:
            raise AssertionError(f"swizzled_bytes({rows}, {D}) = {got}, expected {expected}")
    print(f"PASS nvfp4 sf swizzled-bytes: {len(cases)} checks")


def _check_reshape(ops, rows: int, D: int) -> None:
    torch.manual_seed(0)
    n_blocks = D // 16
    scales = torch.randint(0, 256, (rows, n_blocks), dtype=torch.uint8)
    out = ops.linear_to_swizzled(scales.cuda())
    expected = _reference_swizzle(scales).cuda()
    torch.testing.assert_close(out, expected)
    print(f"PASS linear_to_swizzled rows={rows} D={D}: exact")


def _check_reuse(ops) -> None:
    scales = torch.arange(4 * 64, device="cuda", dtype=torch.uint8).reshape(4, 64)
    out = torch.zeros(_swizzled_bytes(4, 1024), device="cuda", dtype=torch.uint8)
    returned = ops.linear_to_swizzled(scales, out=out)
    if returned is not out:
        raise AssertionError("out tensor must be reused in place")
    print("PASS linear_to_swizzled reuses out")


def _check_invalid(ops) -> None:
    for fn, args, match in [
        (ops.swizzled_bytes, (0, 4096), "rows"),
        (ops.swizzled_bytes, (1, 15), "D"),
    ]:
        try:
            fn(*args)
        except (ValueError, RuntimeError) as exc:
            if match not in str(exc):
                raise AssertionError(f"expected error containing {match!r}, got {exc}")
        else:
            raise AssertionError(f"expected error containing {match!r}")
    print("PASS swizzled-bytes rejects invalid shape")


def run(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    checks = 0
    _check_bytes(ops); checks += 1
    shapes = [(1, 1024), (4, 4096), (33, 4096), (128, 4096), (129, 4096), (16, 12288)]
    if args.mode == "full":
        shapes += [(257, 4096), (64, 16384)]
    for rows, D in shapes:
        _check_reshape(ops, rows, D)
        checks += 1
    _check_reuse(ops); checks += 1
    _check_invalid(ops); checks += 1
    print(f"PASS flashrt-nvfp4 {args.backend} mode={args.mode}: {checks} checks")


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
