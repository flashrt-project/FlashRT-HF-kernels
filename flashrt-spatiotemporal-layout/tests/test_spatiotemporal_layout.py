#!/usr/bin/env python3
"""Correctness tests for flashrt-spatiotemporal-layout."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-spatiotemporal-layout"
REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def ncdhw_to_blc_bf16(self, x, out=None):
        if out is None:
            out = torch.empty((x.shape[0], x.shape[2] * x.shape[3] * x.shape[4], x.shape[1]), device=x.device, dtype=x.dtype)
        self._ops.ncdhw_to_blc_bf16(x, out)
        return out

    def patch_im2col_bf16(self, x, out=None):
        if out is None:
            out = torch.empty((x.shape[0] * 256, 588), device=x.device, dtype=x.dtype)
        self._ops.patch_im2col_bf16(x, out)
        return out

    def time_unshuffle2_bf16(self, x, out=None):
        if out is None:
            out = torch.empty((x.shape[0], x.shape[1] // 2, 2 * x.shape[2], x.shape[3], x.shape[4]), device=x.device, dtype=x.dtype)
        self._ops.time_unshuffle2_bf16(x, out)
        return out

    def add_bias_ncdhw_bf16(self, x, bias):
        self._ops.add_bias_ncdhw_bf16(x, bias)
        return x

    def update_cache2_ncdhw_bf16(self, cur, prev, out=None):
        if out is None:
            out = torch.empty((cur.shape[0], cur.shape[1], 2, cur.shape[3], cur.shape[4]), device=cur.device, dtype=cur.dtype)
        self._ops.update_cache2_ncdhw_bf16(cur, prev, out)
        return out

    def channel_to_space3d_bf16(
        self, x, out_channels, temporal_factor, spatial_factor,
        repeats=1, first_chunk=False, out=None
    ):
        out_t = x.shape[2] * temporal_factor - (
            temporal_factor - 1 if first_chunk else 0
        )
        if out is None:
            out = torch.empty(
                (
                    x.shape[0], out_channels, out_t,
                    x.shape[3] * spatial_factor,
                    x.shape[4] * spatial_factor,
                ),
                device=x.device,
                dtype=x.dtype,
            )
        self._ops.channel_to_space3d_bf16(
            x, out_channels, temporal_factor, spatial_factor,
            repeats, first_chunk, out
        )
        return out

    def pack_causal_cache3_nhwc_bf16(self, previous, current, out=None):
        if out is None:
            out = torch.empty(
                (
                    current.shape[0], current.shape[3], current.shape[4],
                    3 * current.shape[1],
                ),
                device=current.device,
                dtype=current.dtype,
            )
        self._ops.pack_causal_cache3_nhwc_bf16(previous, current, out)
        return out

    def avg_pool3d_channels_bf16(
        self,
        x,
        out_channels,
        factor_t,
        factor_s,
        group_size,
        out=None,
    ):
        if out is None:
            out = torch.empty(
                (
                    x.shape[0],
                    out_channels,
                    (x.shape[2] + factor_t - 1) // factor_t,
                    x.shape[3] // factor_s,
                    x.shape[4] // factor_s,
                ),
                device=x.device,
                dtype=x.dtype,
            )
        self._ops.avg_pool3d_channels_bf16(
            x, out_channels, factor_t, factor_s, group_size, out
        )
        return out

    def ndhwc_to_ncdhw_bf16(self, x, out=None):
        if out is None:
            out = torch.empty(
                (x.shape[0], x.shape[4], x.shape[1], x.shape[2], x.shape[3]),
                device=x.device,
                dtype=x.dtype,
            )
        self._ops.ndhwc_to_ncdhw_bf16(x, out)
        return out

    def ndhwc_to_ncdhw_bias_bf16(self, x, bias, out=None):
        if out is None:
            out = torch.empty(
                (x.shape[0], x.shape[4], x.shape[1], x.shape[2], x.shape[3]),
                device=x.device,
                dtype=x.dtype,
            )
        self._ops.ndhwc_to_ncdhw_bias_bf16(x, bias, out)
        return out

    def ndhwc_to_ncdhw_add_bf16(self, x, residual, out=None):
        if out is None:
            out = torch.empty_like(residual)
        self._ops.ndhwc_to_ncdhw_add_bf16(x, residual, out)
        return out

    def ncdhw_quantize_fp8_static_ndhwc_bf16(self, x, scale, out=None):
        if out is None:
            out = torch.empty(
                (x.shape[0], x.shape[2], x.shape[3], x.shape[4], x.shape[1]),
                device=x.device,
                dtype=torch.float8_e4m3fn,
            )
        self._ops.ncdhw_quantize_fp8_static_ndhwc_bf16(x, scale, out)
        return out

    def upsample2x_quantize_fp8_static_nhwc_bf16(self, x, scale, out=None):
        if out is None:
            out = torch.empty(
                (x.shape[0], 2 * x.shape[2], 2 * x.shape[3], x.shape[1]),
                device=x.device,
                dtype=torch.float8_e4m3fn,
            )
        self._ops.upsample2x_quantize_fp8_static_nhwc_bf16(x, scale, out)
        return out


def _preload_cublaslt() -> None:
    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    library = ctypes.util.find_library("cublasLt")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
    namespace = "flashrt_spatiotemporal_layout_test"
    load(
        name=namespace,
        sources=[
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "spatiotemporal_layout.cu"),
            str(PACKAGE / "csrc" / "bf16_ndhwc_to_ncdhw_transpose.cu"),
            str(PACKAGE / "csrc" / "bf16_quant_fp8_ncdhw_to_ndhwc.cu"),
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
        return importlib.import_module("flashrt_spatiotemporal_layout")
    finally:
        if artifact:
            sys.path.remove(artifact)


def assert_exact(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (got.float() - expected.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    if max_abs != 0.0:
        raise AssertionError(f"{name} failed: max_abs={max_abs}")
    print(f"PASS {name}: max_abs=0")


def ref_cache2(cur: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    out = torch.empty((cur.shape[0], cur.shape[1], 2, cur.shape[3], cur.shape[4]), device=cur.device, dtype=cur.dtype)
    if cur.shape[2] >= 2:
        out.copy_(cur[:, :, -2:, :, :])
    else:
        out[:, :, 0].copy_(prev[:, :, 1])
        out[:, :, 1].copy_(cur[:, :, 0])
    return out


def ref_patch_im2col(x: torch.Tensor) -> torch.Tensor:
    return (
        x.reshape(x.shape[0], 16, 14, 16, 14, 3)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(x.shape[0] * 256, 588)
    )


def ref_channel_to_space3d(
    x: torch.Tensor,
    out_channels: int,
    temporal_factor: int,
    spatial_factor: int,
    repeats: int,
    first_chunk: bool,
) -> torch.Tensor:
    b, in_channels, t, h, w = x.shape
    out_t = t * temporal_factor - (
        temporal_factor - 1 if first_chunk else 0
    )
    out_h = h * spatial_factor
    out_w = w * spatial_factor
    total = b * out_channels * out_t * out_h * out_w
    index = torch.arange(total, device=x.device, dtype=torch.int64)
    ow = index % out_w
    rem = index // out_w
    oh = rem % out_h
    rem //= out_h
    ot = rem % out_t
    rem //= out_t
    oc = rem % out_channels
    batch = rem // out_channels
    full_t = ot + (t * temporal_factor - out_t)
    it = full_t // temporal_factor
    dt = full_t % temporal_factor
    ih = oh // spatial_factor
    dh = oh % spatial_factor
    iw = ow // spatial_factor
    dw = ow % spatial_factor
    subpixel = ((dt * spatial_factor) + dh) * spatial_factor + dw
    channel = (
        oc * temporal_factor * spatial_factor * spatial_factor + subpixel
    ) // repeats
    source = (
        ((((batch * in_channels + channel) * t + it) * h + ih) * w)
        + iw
    )
    return x.flatten()[source].reshape(
        b, out_channels, out_t, out_h, out_w
    )


def ref_avg_pool3d_channels(
    x: torch.Tensor,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    group_size: int,
) -> torch.Tensor:
    pad_t = (-x.shape[2]) % factor_t
    padded = torch.nn.functional.pad(x, (0, 0, 0, 0, pad_t, 0))
    b, c, t, h, w = padded.shape
    folded = (
        padded.view(
            b,
            c,
            t // factor_t,
            factor_t,
            h // factor_s,
            factor_s,
            w // factor_s,
            factor_s,
        )
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .contiguous()
        .view(
            b,
            out_channels,
            group_size,
            t // factor_t,
            h // factor_s,
            w // factor_s,
        )
    )
    return folded.float().mean(dim=2).to(torch.bfloat16)


def run_shape(ops, label: str, shape: tuple[int, int, int, int, int]) -> None:
    b, c, t, h, w = shape
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    got_blc = ops.ncdhw_to_blc_bf16(x)
    exp_blc = x.permute(0, 2, 3, 4, 1).contiguous().view(b, t * h * w, c)
    assert_exact(f"{label}/ncdhw_to_blc", got_blc, exp_blc)

    x2 = torch.randn((b, 2 * c, t, h, w), device="cuda", dtype=torch.bfloat16)
    got_unshuffle = ops.time_unshuffle2_bf16(x2)
    exp_unshuffle = torch.empty((b, c, 2 * t, h, w), device="cuda", dtype=torch.bfloat16)
    exp_unshuffle[:, :, 0::2] = x2[:, :c]
    exp_unshuffle[:, :, 1::2] = x2[:, c:]
    assert_exact(f"{label}/time_unshuffle2", got_unshuffle, exp_unshuffle)

    bias = torch.randn((c,), device="cuda", dtype=torch.bfloat16)
    x_bias = x.clone()
    got_bias = ops.add_bias_ncdhw_bf16(x_bias, bias)
    exp_bias = (x.float() + bias.float().view(1, c, 1, 1, 1)).to(torch.bfloat16)
    assert_exact(f"{label}/add_bias_ncdhw", got_bias, exp_bias)

    prev = torch.randn((b, c, 2, h, w), device="cuda", dtype=torch.bfloat16)
    got_cache = ops.update_cache2_ncdhw_bf16(x, prev)
    assert_exact(f"{label}/update_cache2", got_cache, ref_cache2(x, prev))

    x_t1 = torch.randn((b, c, 1, h, w), device="cuda", dtype=torch.bfloat16)
    got_cache_t1 = ops.update_cache2_ncdhw_bf16(x_t1, prev)
    assert_exact(f"{label}/update_cache2_t1", got_cache_t1, ref_cache2(x_t1, prev))

    packed_cache = ops.pack_causal_cache3_nhwc_bf16(prev, x_t1)
    expected_packed_cache = torch.cat(
        (prev[:, :, 0], prev[:, :, 1], x_t1[:, :, 0]), dim=1
    ).permute(0, 2, 3, 1).contiguous()
    assert_exact(
        f"{label}/pack_causal_cache3_nhwc",
        packed_cache,
        expected_packed_cache,
    )

    if h % 2 == 0 and w % 2 == 0:
        factor_t, factor_s, group_size = 2, 2, 8
        out_channels = c
        got_pool = ops.avg_pool3d_channels_bf16(
            x,
            out_channels,
            factor_t,
            factor_s,
            group_size,
        )
        expected_pool = ref_avg_pool3d_channels(
            x,
            out_channels,
            factor_t,
            factor_s,
            group_size,
        )
        torch.testing.assert_close(
            got_pool, expected_pool, rtol=0.0, atol=0.015625
        )
        print(f"PASS {label}/avg_pool3d_channels")

    ndhwc = x.permute(0, 2, 3, 4, 1).contiguous()
    assert_exact(
        f"{label}/ndhwc_to_ncdhw",
        ops.ndhwc_to_ncdhw_bf16(ndhwc),
        x,
    )
    got_bias_layout = ops.ndhwc_to_ncdhw_bias_bf16(ndhwc, bias)
    assert_exact(
        f"{label}/ndhwc_to_ncdhw_bias",
        got_bias_layout,
        exp_bias,
    )
    residual = torch.randn_like(x)
    got_add_layout = ops.ndhwc_to_ncdhw_add_bf16(ndhwc, residual)
    exp_add_layout = (x.float() + residual.float()).to(torch.bfloat16)
    assert_exact(
        f"{label}/ndhwc_to_ncdhw_add",
        got_add_layout,
        exp_add_layout,
    )

    if c % 4 == 0:
        scale = 0.03125
        got_fp8 = ops.ncdhw_quantize_fp8_static_ndhwc_bf16(x, scale)
        exp_fp8 = (
            (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
        )
        if not torch.equal(got_fp8.view(torch.uint8), exp_fp8.view(torch.uint8)):
            raise AssertionError(f"{label}/ncdhw_quantize_fp8 is not bitwise exact")
        print(f"PASS {label}/ncdhw_quantize_fp8: bitwise exact")

        x2d = x[:, :, 0].contiguous()
        got_up = ops.upsample2x_quantize_fp8_static_nhwc_bf16(x2d, scale)
        exp_up = (
            torch.nn.functional.interpolate(x2d.float(), scale_factor=2.0, mode="nearest")
            .div(scale)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        if not torch.equal(got_up.view(torch.uint8), exp_up.view(torch.uint8)):
            raise AssertionError(f"{label}/upsample2x_quantize_fp8 is not bitwise exact")
        print(f"PASS {label}/upsample2x_quantize_fp8: bitwise exact")


def run_patch_shape(ops, num_views: int) -> None:
    x = torch.randn((num_views, 224, 224, 3), device="cuda", dtype=torch.bfloat16)
    got = ops.patch_im2col_bf16(x)
    assert_exact(f"patch_nv{num_views}/patch_im2col", got, ref_patch_im2col(x))


def run_channel_to_space_shapes(ops) -> None:
    for b, t, h, w, first_chunk in (
        (1, 3, 4, 5, False),
        (1, 3, 4, 5, True),
        (2, 1, 3, 7, False),
    ):
        x = torch.randn(
            (b, 8, t, h, w), device="cuda", dtype=torch.bfloat16
        )
        got = ops.channel_to_space3d_bf16(
            x, 2, 2, 2, repeats=2, first_chunk=first_chunk
        )
        expected = ref_channel_to_space3d(
            x, 2, 2, 2, repeats=2, first_chunk=first_chunk
        )
        assert_exact(
            f"channel_to_space/b{b}_t{t}_h{h}_w{w}_first{first_chunk}",
            got,
            expected,
        )


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(59)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    shapes = {
        "small": (1, 8, 4, 8, 8),
        "latent_16": (1, 16, 8, 32, 32),
        "latent_64": (1, 64, 4, 32, 32),
    }
    if args.mode == "smoke":
        shapes = {"small": shapes["small"]}
    for label, shape in shapes.items():
        run_shape(ops, label, shape)
    run_patch_shape(ops, 2)
    run_channel_to_space_shapes(ops)

    x = torch.randn(
        (1, 8, 5, 8, 8), device="cuda", dtype=torch.bfloat16
    )

    def invoke(value):
        return ops.avg_pool3d_channels_bf16(value, 8, 2, 2, 8)

    eager = invoke(x)
    compiled = torch.compile(invoke, fullgraph=True)(x)
    torch.testing.assert_close(compiled, eager, rtol=0.0, atol=0.0)
    print("PASS avg_pool3d_channels torch.compile fullgraph")

    x_layout = torch.randn(
        (1, 64, 4, 16, 16), device="cuda", dtype=torch.bfloat16
    )

    def invoke_layout(value):
        quant = ops.ncdhw_quantize_fp8_static_ndhwc_bf16(value, 0.03125)
        return quant, ops.ndhwc_to_ncdhw_bf16(
            value.permute(0, 2, 3, 4, 1).contiguous()
        )

    eager_quant, eager_layout = invoke_layout(x_layout)
    compiled_quant, compiled_layout = torch.compile(
        invoke_layout, fullgraph=True
    )(x_layout)
    if not torch.equal(
        compiled_quant.view(torch.uint8), eager_quant.view(torch.uint8)
    ):
        raise AssertionError("compiled FP8 layout output is not bitwise exact")
    assert_exact("layout producer torch.compile", compiled_layout, eager_layout)

    prev = torch.randn(
        (1, 64, 2, 8, 8), device="cuda", dtype=torch.bfloat16
    )
    current = torch.randn(
        (1, 64, 1, 8, 8), device="cuda", dtype=torch.bfloat16
    )

    def invoke_cache_and_upsample(previous, value):
        packed = ops.pack_causal_cache3_nhwc_bf16(previous, value)
        upsampled = ops.channel_to_space3d_bf16(
            value, 16, 1, 2, repeats=1, first_chunk=False
        )
        return packed, upsampled

    eager_cache, eager_up = invoke_cache_and_upsample(prev, current)
    compiled_cache, compiled_up = torch.compile(
        invoke_cache_and_upsample, fullgraph=True
    )(prev, current)
    assert_exact("cache pack torch.compile", compiled_cache, eager_cache)
    assert_exact("channel-to-space torch.compile", compiled_up, eager_up)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
