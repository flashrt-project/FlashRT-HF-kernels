import torch

from kernels.benchmark import Benchmark


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
        for blk in range(n_blocks):
            cb = blk // 4
            ci = blk % 4
            super_idx = rb * n_col_super + cb
            inner_off = (ri % 32) * 16 + (ri // 32) * 4 + ci
            out[super_idx * 512 + inner_off] = src[row, blk]
    return out.to(scales.device)


class Nvfp4ScaleFactorReshapeBenchmark(Benchmark):
    seed = 7

    def _setup_shape(self, rows: int, D: int) -> None:
        self.scales = torch.randint(
            0,
            256,
            (rows, D // 16),
            device=self.device,
            dtype=torch.uint8,
        )
        n_col_super = ((D // 16) + 3) // 4
        self.out = torch.zeros(
            ((rows + 127) // 128) * n_col_super * 512,
            device=self.device,
            dtype=torch.uint8,
        )

    def _reference(self):
        return _reference_swizzle(self.scales)

    def setup_rows1_d4096(self):
        self._setup_shape(1, 4096)

    def benchmark_rows1_d4096(self):
        self.kernel.nvfp4_sf_linear_to_swizzled(self.scales, out=self.out)

    def verify_rows1_d4096(self):
        return self._reference()

    def setup_rows16_d12288(self):
        self._setup_shape(16, 12288)

    def benchmark_rows16_d12288(self):
        self.kernel.nvfp4_sf_linear_to_swizzled(self.scales, out=self.out)

    def verify_rows16_d12288(self):
        return self._reference()

    def setup_rows64_d16384(self):
        self._setup_shape(64, 16384)

    def benchmark_rows64_d16384(self):
        self.kernel.nvfp4_sf_linear_to_swizzled(self.scales, out=self.out)

    def verify_rows64_d16384(self):
        return self._reference()

    def setup_rows128_d4096(self):
        self._setup_shape(128, 4096)

    def benchmark_rows128_d4096(self):
        self.kernel.nvfp4_sf_linear_to_swizzled(self.scales, out=self.out)

    def verify_rows128_d4096(self):
        return self._reference()

    def setup_rows129_d4096(self):
        self._setup_shape(129, 4096)

    def benchmark_rows129_d4096(self):
        self.kernel.nvfp4_sf_linear_to_swizzled(self.scales, out=self.out)

    def verify_rows129_d4096(self):
        return self._reference()
