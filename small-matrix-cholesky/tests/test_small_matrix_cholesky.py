#!/usr/bin/env python3
"""Correctness tests for small-matrix-cholesky."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source_loader import load_source_ops  # noqa: E402


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("small_matrix_cholesky")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_spd(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    n = shape[-1]
    generator = torch.Generator(device=device).manual_seed(
        1701 + n + sum(shape[:-2])
    )
    x = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ) / n**0.5
    identity = torch.eye(n, device=device, dtype=torch.float32)
    return (x @ x.transpose(-1, -2) + 0.5 * identity).contiguous()


def check_factor(name: str, input: torch.Tensor, output: torch.Tensor) -> None:
    if output.shape != input.shape:
        raise AssertionError(f"{name}: shape mismatch")
    if output.dtype != torch.float32 or output.device != input.device:
        raise AssertionError(f"{name}: dtype/device mismatch")
    if not torch.equal(output, torch.tril(output)):
        raise AssertionError(f"{name}: upper triangle is not exactly zero")
    if not bool((output.diagonal(dim1=-2, dim2=-1) > 0).all()):
        raise AssertionError(f"{name}: diagonal must be positive")

    reconstruction = output @ output.transpose(-1, -2)
    numerator = torch.linalg.matrix_norm(
        reconstruction - input, ord="fro"
    )
    denominator = torch.linalg.matrix_norm(input, ord="fro")
    max_relative = float((numerator / denominator).max().item())
    if max_relative > 3e-4:
        raise AssertionError(
            f"{name}: reconstruction relative error={max_relative:.8e}"
        )

    reference = torch.linalg.cholesky(input)
    torch.testing.assert_close(
        output,
        reference,
        rtol=5e-4,
        atol=2e-4,
        msg=lambda message: f"{name}: {message}",
    )


def expect_runtime_error(name: str, fn) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError(f"{name}: expected RuntimeError")


def run_validation_cases(ops, device: torch.device) -> int:
    valid = make_spd((3, 32, 32), device)
    expect_runtime_error(
        "dtype",
        lambda: ops.cholesky_small_fp32(valid.to(torch.float16)),
    )
    expect_runtime_error(
        "unsupported-order",
        lambda: ops.cholesky_small_fp32(
            torch.eye(48, device=device).expand(2, 48, 48).contiguous()
        ),
    )
    expect_runtime_error(
        "non-square",
        lambda: ops.cholesky_small_fp32(
            torch.empty(2, 32, 64, device=device)
        ),
    )
    expect_runtime_error(
        "non-contiguous",
        lambda: ops.cholesky_small_fp32(valid.transpose(-1, -2)),
    )
    expect_runtime_error(
        "alias",
        lambda: ops.cholesky_small_fp32(valid, out=valid),
    )
    expect_runtime_error(
        "wrong-output-shape",
        lambda: ops.cholesky_small_fp32(
            valid,
            out=torch.empty(2, 32, 32, device=device),
        ),
    )
    expect_runtime_error(
        "cpu",
        lambda: ops.cholesky_small_fp32(valid.cpu()),
    )
    return 7


def run(ops, mode: str) -> int:
    device = torch.device("cuda", 0)
    shapes = [(7, 32, 32), (5, 64, 64), (3, 128, 128)]
    if mode == "full":
        shapes += [
            (2, 3, 32, 32),
            (4096, 32, 32),
            (1024, 64, 64),
            (256, 128, 128),
        ]

    count = 0
    with torch.cuda.device(device):
        for shape in shapes:
            input = make_spd(shape, device)
            preallocated = torch.empty_like(input)
            output = ops.cholesky_small_fp32(input, out=preallocated)
            torch.cuda.synchronize(device)
            if output.data_ptr() != preallocated.data_ptr():
                raise AssertionError("the out tensor was not returned")
            check_factor(f"shape={shape}", input, output)
            count += 1
        count += run_validation_cases(ops, device)

    if torch.cuda.device_count() > 1:
        second = torch.device("cuda", 1)
        with torch.cuda.device(second):
            input = make_spd((4, 128, 128), second)
            output = ops.cholesky_small_fp32(input)
            torch.cuda.synchronize(second)
            check_factor("non-default-device", input, output)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["source", "installed"], default="source"
    )
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--registration-include", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = (
        load_source_ops(args.registration_include)
        if args.backend == "source"
        else load_installed_ops(args.artifact)
    )
    count = run(ops, args.mode)
    print(
        f"small-matrix-cholesky {args.backend} {args.mode}: "
        f"passed {count}/{count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
