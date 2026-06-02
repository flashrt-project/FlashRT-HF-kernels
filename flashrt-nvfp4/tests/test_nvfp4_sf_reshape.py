import pytest
import torch

from flashrt_nvfp4 import (
    nvfp4_sf_linear_to_swizzled,
    nvfp4_sf_swizzled_bytes,
)


def _reference_swizzle(scales: torch.Tensor) -> torch.Tensor:
    rows, n_blocks = scales.shape
    n_col_super = (n_blocks + 3) // 4
    out = torch.zeros(
        ((rows + 127) // 128) * n_col_super * 512,
        dtype=torch.uint8,
    )
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


@pytest.mark.parametrize(
    ("rows", "D"),
    [
        (1, 1024),
        (2, 4096),
        (31, 4096),
        (32, 4096),
        (33, 4096),
        (127, 4096),
        (128, 4096),
        (129, 4096),
        (16, 12288),
        (64, 16384),
    ],
)
def test_nvfp4_sf_swizzled_bytes(rows, D):
    n_blocks = D // 16
    expected = ((rows + 127) // 128) * ((n_blocks + 3) // 4) * 512
    assert nvfp4_sf_swizzled_bytes(rows, D) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("rows", "D"),
    [
        (1, 1024),
        (4, 4096),
        (33, 4096),
        (128, 4096),
        (129, 4096),
        (16, 12288),
    ],
)
def test_nvfp4_sf_linear_to_swizzled(rows, D):
    torch.manual_seed(0)
    n_blocks = D // 16
    scales_cpu = torch.randint(0, 256, (rows, n_blocks), dtype=torch.uint8)
    scales = scales_cpu.cuda()

    out = nvfp4_sf_linear_to_swizzled(scales)
    expected = _reference_swizzle(scales_cpu).cuda()

    torch.testing.assert_close(out, expected)


def test_nvfp4_sf_swizzled_bytes_rejects_invalid_shape():
    with pytest.raises(ValueError, match="rows"):
        nvfp4_sf_swizzled_bytes(0, 4096)
    with pytest.raises(ValueError, match="D"):
        nvfp4_sf_swizzled_bytes(1, 15)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_sf_linear_to_swizzled_reuses_out():
    scales = torch.arange(4 * 64, device="cuda", dtype=torch.uint8).reshape(4, 64)
    out = torch.zeros(
        (nvfp4_sf_swizzled_bytes(4, 1024),),
        device="cuda",
        dtype=torch.uint8,
    )

    returned = nvfp4_sf_linear_to_swizzled(scales, out=out)

    assert returned is out
