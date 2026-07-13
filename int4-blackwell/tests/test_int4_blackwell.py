import pytest
import torch

import int4_blackwell


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in {(12, 0), (12, 1)},
    reason="int4-blackwell requires SM120 or SM121",
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
    got = int4_blackwell.codebook_probe(mode)
    torch.testing.assert_close(
        got, torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0
    )


def test_mma_probe_launches():
    scratch = torch.empty((1, 256), device="cuda", dtype=torch.float32)
    output = int4_blackwell.mma_probe(iterations=16, blocks=1, out=scratch)
    assert output is scratch
    assert output.shape == (1, 256)
    torch.cuda.synchronize()
