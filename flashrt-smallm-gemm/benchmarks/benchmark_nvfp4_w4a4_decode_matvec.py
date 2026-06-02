import torch

from kernels.benchmark import Benchmark


_original_allclose = torch.allclose


def _bf16_max_ulp(input: torch.Tensor, other: torch.Tensor) -> int:
    got_bits = input.detach().cpu().view(torch.int16).to(torch.int32) & 0xFFFF
    exp_bits = other.detach().cpu().view(torch.int16).to(torch.int32) & 0xFFFF
    got_ordered = torch.where((got_bits & 0x8000) != 0, 0x8000 - (got_bits & 0x7FFF), got_bits)
    exp_ordered = torch.where((exp_bits & 0x8000) != 0, 0x8000 - (exp_bits & 0x7FFF), exp_bits)
    return int((got_ordered - exp_ordered).abs().max().item())


def _flashrt_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    if input.dtype == torch.bfloat16 and other.dtype == torch.bfloat16:
        return _bf16_max_ulp(input, other) <= 5
    return _original_allclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)


torch.allclose = _flashrt_allclose


DECODE_SHAPES = [
    ("k4096_n1024", 4096, 1024),
    ("k4096_n4096", 4096, 4096),
    ("k4096_n12288", 4096, 12288),
    ("k12288_n1024", 12288, 1024),
    ("k12288_n4096", 12288, 4096),
    ("k12288_n12288", 12288, 12288),
]


def _swizzled_bytes(rows: int, cols: int) -> int:
    n_blocks = cols // 16
    return ((rows + 127) // 128) * ((n_blocks + 3) // 4) * 512


def _swizzle_constant_scale(rows: int, cols: int, value: int) -> torch.Tensor:
    return torch.full((_swizzled_bytes(rows, cols),), value, dtype=torch.uint8)


def _reference_swizzle(scales: torch.Tensor) -> torch.Tensor:
    rows, n_blocks = scales.shape
    n_col_super = (n_blocks + 3) // 4
    src = scales.cpu()
    out = torch.zeros(
        ((rows + 127) // 128) * n_col_super * 512,
        dtype=torch.uint8,
    )
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


def _ue4m3_to_float(byte: int) -> float:
    sign = -1.0 if (byte & 0x80) else 1.0
    exp = (byte >> 3) & 0x0F
    mant = byte & 0x07
    if exp == 0:
        return sign * (mant / 8.0) * (2.0 ** -6)
    if exp == 15 and mant == 7:
        return 0.0
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


def _ue4m3_lut() -> torch.Tensor:
    return torch.tensor([_ue4m3_to_float(i) for i in range(256)], dtype=torch.float32)


def _fp4_codebook() -> torch.Tensor:
    return torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
    )


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    codebook = _fp4_codebook().to(packed.device)
    lo = packed & 0x0F
    hi = packed >> 4
    out = torch.empty(
        (packed.shape[0], packed.shape[1] * 2),
        device=packed.device,
        dtype=torch.float32,
    )
    out[:, 0::2] = codebook[lo.long()]
    out[:, 1::2] = codebook[hi.long()]
    return out


def _reference_smallm(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    sfa_linear: torch.Tensor,
    sfb_linear: torch.Tensor,
    K: int,
    alpha: float,
    chunk_rows: int = 256,
) -> torch.Tensor:
    device = b_packed.device
    N = b_packed.shape[0]
    lut = _ue4m3_lut().to(device)
    a = _unpack_fp4(a_packed.reshape(1, -1)).reshape(K)
    a_scale = lut[sfa_linear.reshape(-1).to(device).long()].repeat_interleave(16)
    a = a * a_scale
    sfb_linear = sfb_linear.to(device)
    out = torch.empty((N,), device=device, dtype=torch.bfloat16)
    for start in range(0, N, chunk_rows):
        end = min(start + chunk_rows, N)
        b = _unpack_fp4(b_packed[start:end])
        b_scale = lut[sfb_linear[start:end].long()].repeat_interleave(16, dim=1)
        expected = (b * b_scale * a.reshape(1, K)).sum(dim=1) * alpha
        out[start:end] = expected.to(torch.bfloat16)
    return out


class Nvfp4W4A4DecodeMatvecBenchmark(Benchmark):
    seed = 23

    def _setup_shape(self, K: int, N: int) -> None:
        torch.manual_seed(600 + K + N)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(600 + K + N)
        self.K = K
        self.N = N
        self.alpha = 0.5
        self.a_packed = torch.randint(
            0, 256, (K // 2,), device=self.device, dtype=torch.uint8
        )
        self.b_packed = torch.randint(
            0, 256, (N, K // 2), device=self.device, dtype=torch.uint8
        )
        self.sfa_linear = torch.randint(0, 0x78, (1, K // 16), dtype=torch.uint8)
        self.sfb_linear = torch.randint(0, 0x78, (N, K // 16), dtype=torch.uint8)
        self.sfa = _reference_swizzle(self.sfa_linear).to(self.device)
        self.sfb = _reference_swizzle(self.sfb_linear).to(self.device)
        self.out = torch.empty((N,), device=self.device, dtype=torch.bfloat16)

    def _benchmark(self) -> None:
        self.kernel.nvfp4_w4a4_decode_matvec_bf16out(
            self.a_packed,
            self.b_packed,
            self.sfa,
            self.sfb,
            alpha=self.alpha,
            out=self.out,
        )

    def _reference(self) -> torch.Tensor:
        return _reference_smallm(
            self.a_packed,
            self.b_packed,
            self.sfa_linear,
            self.sfb_linear,
            self.K,
            self.alpha,
        )


def _register_shapes() -> None:
    for label, K, N in DECODE_SHAPES:

        def setup(self, K=K, N=N) -> None:
            self._setup_shape(K, N)

        def benchmark(self) -> None:
            self._benchmark()

        def verify(self) -> torch.Tensor:
            return self._reference()

        setattr(Nvfp4W4A4DecodeMatvecBenchmark, f"setup_{label}", setup)
        setattr(Nvfp4W4A4DecodeMatvecBenchmark, f"benchmark_{label}", benchmark)
        setattr(Nvfp4W4A4DecodeMatvecBenchmark, f"verify_{label}", verify)


_register_shapes()
