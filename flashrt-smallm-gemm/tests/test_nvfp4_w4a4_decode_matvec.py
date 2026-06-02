import pytest
import torch

from flashrt_smallm_gemm import nvfp4_w4a4_decode_matvec_bf16out


def _swizzled_bytes(rows: int, D: int) -> int:
    n_blocks = D // 16
    return ((rows + 127) // 128) * ((n_blocks + 3) // 4) * 512


def _swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    rows, n_blocks = scales.shape
    n_col_super = (n_blocks + 3) // 4
    out = torch.zeros(
        (_swizzled_bytes(rows, n_blocks * 16),),
        dtype=torch.uint8,
    )
    src = scales.cpu()
    for row in range(rows):
        rb = row // 128
        ri = row % 128
        for block in range(n_blocks):
            cb = block // 4
            ci = block % 4
            super_idx = rb * n_col_super + cb
            inner_off = (ri % 32) * 16 + (ri // 32) * 4 + ci
            out[super_idx * 512 + inner_off] = src[row, block]
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("K", [4096, 12288])
def test_nvfp4_w4a4_decode_matvec_constant_inputs(K):
    torch.manual_seed(0)
    N = 16
    alpha = 0.5
    a_packed = torch.full((K // 2,), 0x11, device="cuda", dtype=torch.uint8)
    b_packed = torch.full((N, K // 2), 0x11, device="cuda", dtype=torch.uint8)
    sfa = _swizzle_scales(torch.full((1, K // 16), 0x38, dtype=torch.uint8)).cuda()
    sfb = _swizzle_scales(torch.full((N, K // 16), 0x38, dtype=torch.uint8)).cuda()

    out = nvfp4_w4a4_decode_matvec_bf16out(
        a_packed,
        b_packed,
        sfa,
        sfb,
        alpha=alpha,
    )

    expected = torch.full((N,), K * 0.25 * alpha, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_w4a4_decode_matvec_reuses_out():
    K = 4096
    N = 8
    a_packed = torch.full((K // 2,), 0x11, device="cuda", dtype=torch.uint8)
    b_packed = torch.full((N, K // 2), 0x11, device="cuda", dtype=torch.uint8)
    sfa = _swizzle_scales(torch.full((1, K // 16), 0x38, dtype=torch.uint8)).cuda()
    sfb = _swizzle_scales(torch.full((N, K // 16), 0x38, dtype=torch.uint8)).cuda()
    out = torch.empty((N,), device="cuda", dtype=torch.bfloat16)

    returned = nvfp4_w4a4_decode_matvec_bf16out(
        a_packed,
        b_packed,
        sfa,
        sfb,
        out=out,
    )

    assert returned is out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_w4a4_decode_matvec_rejects_unsupported_k():
    K = 8192
    N = 8
    a_packed = torch.full((K // 2,), 0x11, device="cuda", dtype=torch.uint8)
    b_packed = torch.full((N, K // 2), 0x11, device="cuda", dtype=torch.uint8)
    sfa = torch.zeros((_swizzled_bytes(1, K),), device="cuda", dtype=torch.uint8)
    sfb = torch.zeros((_swizzled_bytes(N, K),), device="cuda", dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="K=4096 and K=12288"):
        nvfp4_w4a4_decode_matvec_bf16out(a_packed, b_packed, sfa, sfb)
