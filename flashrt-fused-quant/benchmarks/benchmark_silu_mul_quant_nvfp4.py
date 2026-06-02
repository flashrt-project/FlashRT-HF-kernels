import torch

from kernels.benchmark import Benchmark


SILU_QUANT_SHAPES = [
    ("decode_r1_h4096", 1, 4096),
    ("decode_r2_h4096", 2, 4096),
    ("decode_r4_h4096", 4, 4096),
    ("decode_r8_h4096", 8, 4096),
    ("decode_r1_h8192", 1, 8192),
    ("decode_r4_h8192", 4, 8192),
    ("decode_r8_h8192", 8, 8192),
    ("decode_r1_h12288", 1, 12288),
    ("decode_r4_h12288", 4, 12288),
    ("decode_r8_h12288", 8, 12288),
    ("decode_r1_h16384", 1, 16384),
    ("decode_r4_h16384", 4, 16384),
    ("decode_r8_h16384", 8, 16384),
    ("small_r16_h4096", 16, 4096),
    ("small_r32_h4096", 32, 4096),
    ("small_r16_h8192", 16, 8192),
    ("small_r32_h8192", 32, 8192),
    ("small_r16_h12288", 16, 12288),
    ("small_r32_h12288", 32, 12288),
    ("small_r16_h16384", 16, 16384),
    ("small_r32_h16384", 32, 16384),
    ("prefill_r64_h4096", 64, 4096),
    ("prefill_r128_h4096", 128, 4096),
    ("prefill_r256_h4096", 256, 4096),
    ("video_r1024_h4096", 1024, 4096),
    ("video_r2520_h4096", 2520, 4096),
    ("prefill_r64_h8192", 64, 8192),
    ("prefill_r128_h8192", 128, 8192),
    ("prefill_r256_h8192", 256, 8192),
    ("video_r1024_h8192", 1024, 8192),
    ("video_r2520_h8192", 2520, 8192),
    ("prefill_r64_h12288", 64, 12288),
    ("prefill_r128_h12288", 128, 12288),
    ("prefill_r256_h12288", 256, 12288),
    ("video_r1024_h12288", 1024, 12288),
    ("video_r2520_h12288", 2520, 12288),
]


def _scale_bytes(rows: int, cols: int) -> int:
    n_blocks = cols // 16
    return ((rows + 127) // 128) * ((n_blocks + 3) // 4) * 512


class SiluMulNvfp4SplitBenchmark(Benchmark):
    seed = 17

    def _setup_shape(self, rows: int, cols: int) -> None:
        self.gate = torch.randn(
            (rows, cols), device=self.device, dtype=torch.bfloat16
        ).contiguous()
        self.up = torch.randn(
            (rows, cols), device=self.device, dtype=torch.bfloat16
        ).contiguous()
        self.packed = torch.empty(
            (rows, cols // 2), device=self.device, dtype=torch.uint8
        )
        self.scales = torch.zeros(
            (_scale_bytes(rows, cols),), device=self.device, dtype=torch.uint8
        )
        self.out = self.packed

    def _benchmark(self) -> None:
        self.kernel.silu_mul_quant_nvfp4_swizzled_bf16(
            self.gate,
            self.up,
            packed=self.packed,
            scales=self.scales,
        )


class SiluMulNvfp4MergedBenchmark(Benchmark):
    seed = 19

    def _setup_shape(self, rows: int, cols: int) -> None:
        gate = torch.randn(
            (rows, cols), device=self.device, dtype=torch.bfloat16
        ).contiguous()
        up = torch.randn((rows, cols), device=self.device, dtype=torch.bfloat16)
        self.merged = torch.cat([gate, up], dim=1).contiguous()
        self.packed = torch.empty(
            (rows, cols // 2), device=self.device, dtype=torch.uint8
        )
        self.scales = torch.zeros(
            (_scale_bytes(rows, cols),), device=self.device, dtype=torch.uint8
        )
        self.out = self.packed

    def _benchmark(self) -> None:
        self.kernel.silu_mul_merged_quant_nvfp4_swizzled_bf16(
            self.merged,
            packed=self.packed,
            scales=self.scales,
        )


def _register_shapes() -> None:
    for label, rows, cols in SILU_QUANT_SHAPES:

        def setup_split(self, rows=rows, cols=cols) -> None:
            self._setup_shape(rows, cols)

        def benchmark_split(self) -> None:
            self._benchmark()

        setattr(SiluMulNvfp4SplitBenchmark, f"setup_{label}", setup_split)
        setattr(SiluMulNvfp4SplitBenchmark, f"benchmark_{label}", benchmark_split)

        def setup_merged(self, rows=rows, cols=cols) -> None:
            self._setup_shape(rows, cols)

        def benchmark_merged(self) -> None:
            self._benchmark()

        setattr(SiluMulNvfp4MergedBenchmark, f"setup_{label}", setup_merged)
        setattr(SiluMulNvfp4MergedBenchmark, f"benchmark_{label}", benchmark_merged)


_register_shapes()
