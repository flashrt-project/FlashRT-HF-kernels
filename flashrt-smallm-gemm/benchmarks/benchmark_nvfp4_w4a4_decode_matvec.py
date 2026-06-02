import torch

from kernels.benchmark import Benchmark


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


class Nvfp4W4A4DecodeMatvecBenchmark(Benchmark):
    seed = 23

    def _setup_shape(self, K: int, N: int) -> None:
        self.K = K
        self.N = N
        self.alpha = 0.5
        self.a_packed = torch.full(
            (K // 2,), 0x11, device=self.device, dtype=torch.uint8
        )
        self.b_packed = torch.full(
            (N, K // 2), 0x11, device=self.device, dtype=torch.uint8
        )
        self.sfa = _swizzle_constant_scale(1, K, 0x38).to(self.device)
        self.sfb = _swizzle_constant_scale(N, K, 0x38).to(self.device)
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
        return torch.full(
            (self.N,),
            self.K * 0.25 * self.alpha,
            device=self.device,
            dtype=torch.bfloat16,
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
