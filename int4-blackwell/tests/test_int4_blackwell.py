import pytest
import torch

import int4_blackwell


SUPPORTED = {(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}
CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None

pytestmark = pytest.mark.skipif(
    CAPABILITY not in SUPPORTED,
    reason="int4-blackwell requires SM100, SM103, SM110, SM120, or SM121",
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("e2m1", [0, .25, .5, .75, 1, 1.5, 2, 3, 0, -.25, -.5, -.75, -1, -1.5, -2, -3]),
        ("a", [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 0, -.5, -1, -1.5, -2, -2.5, -3, -3.5]),
        ("b", [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]),
        ("ab", [0, 1, 2, 3, 4, 5, 6, 7, 0, -1, -2, -3, -4, -5, -6, -7]),
    ],
)
def test_codebook(mode, expected):
    if CAPABILITY in {(10, 0), (10, 3), (11, 0)} and mode != "ab":
        pytest.skip("tcgen05 currently validates the native INT4 x INT4 mode")
    got = int4_blackwell.codebook_probe(mode)
    torch.testing.assert_close(
        got, torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0
    )


def test_mma_probe_launches():
    if CAPABILITY in {(10, 0), (10, 3), (11, 0)}:
        pytest.skip("register-resident mma_probe is specific to SM12x OMMA")
    scratch = torch.empty((1, 256), device="cuda", dtype=torch.float32)
    output = int4_blackwell.mma_probe(iterations=16, blocks=1, out=scratch)
    assert output is scratch
    assert output.shape == (1, 256)
    torch.cuda.synchronize()


@pytest.mark.parametrize("shape", [(128, 128, 128), (128, 256, 256), (256, 128, 128)])
def test_tcgen05_int4_gemm_scale_one(shape):
    if CAPABILITY not in {(10, 0), (10, 3), (11, 0)}:
        pytest.skip("tcgen05 GEMM is specific to SM100, SM103, and SM110")
    torch.manual_seed(20260713)
    m, n, k = shape
    a = torch.randint(-2, 3, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-2, 3, (n, k), device="cuda", dtype=torch.int8)

    def pack(values):
        codes = torch.where(values >= 0, values, -values + 8).to(torch.uint8)
        return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()

    # UE4M3 bit pattern 0x38 is one. Constant scales are independent of the
    # physical CUTLASS scale-factor permutation and isolate GEMM correctness.
    sfa = torch.full((m * k,), 0x38, device="cuda", dtype=torch.uint8)
    sfb = torch.full((n * k,), 0x38, device="cuda", dtype=torch.uint8)
    actual = int4_blackwell.tcgen05_int4_gemm_bf16(
        pack(a), sfa, pack(b), sfb
    )
    expected = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
