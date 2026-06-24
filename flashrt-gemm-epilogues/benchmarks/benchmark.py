import torch

from kernels.benchmark import Benchmark


_original_allclose = torch.allclose


def _fp8_dtype() -> torch.dtype:
    if torch.version.hip is not None and hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _fp8_max() -> float:
    return 240.0 if torch.version.hip is not None else 448.0


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype == torch.float8_e4m3fn or (
        hasattr(torch, "float8_e4m3fnuz") and dtype == torch.float8_e4m3fnuz
    )


def _flashrt_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    if _is_fp8_dtype(input.dtype) or _is_fp8_dtype(other.dtype):
        return _original_allclose(
            input.float(),
            other.float(),
            rtol=0,
            atol=0,
            equal_nan=equal_nan,
        )
    return _original_allclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)


torch.allclose = _flashrt_allclose


QUANT_SHAPES = [
    ("decode_m1", 1, 4096),
    ("decode_m2", 2, 4096),
    ("decode_m4", 4, 4096),
    ("decode_m8", 8, 4096),
    ("small_m16", 16, 4096),
    ("small_m32", 32, 4096),
    ("prefill_m64", 64, 4096),
    ("prefill_m128", 128, 4096),
    ("prefill_m256", 256, 4096),
    ("wide_n8192_m16", 16, 8192),
    ("wide_n8192_m128", 128, 8192),
    ("vla_n12288_m16", 16, 12288),
    ("vla_n12288_m64", 64, 12288),
    ("vla_n16384_m16", 16, 16384),
    ("vla_n16384_m64", 64, 16384),
]


class BiasGeluFp8QuantizeBenchmark(Benchmark):
    seed = 1

    def _setup_shape(self, m: int, n: int) -> None:
        self.input = torch.randn((m, n), device=self.device, dtype=torch.bfloat16)
        self.bias = torch.randn((n,), device=self.device, dtype=torch.bfloat16)
        self.scale = torch.tensor([0.25], device=self.device, dtype=torch.float32)
        self.out = torch.empty_like(self.input, dtype=_fp8_dtype())

    def _reference(self) -> torch.Tensor:
        y = self.input.float() + self.bias.float()
        y = torch.nn.functional.gelu(y, approximate="tanh")
        limit = _fp8_max()
        y = torch.clamp(y / self.scale.float(), -limit, limit)
        return y.to(_fp8_dtype())

    def setup_decode_m1(self) -> None:
        self._setup_shape(1, 4096)

    def benchmark_decode_m1(self) -> None:
        self.kernel.bias_gelu_quantize_fp8_static_bf16(
            self.input, self.bias, self.scale, out=self.out
        )

    def verify_decode_m1(self) -> torch.Tensor:
        return self._reference()

    def setup_decode_m8(self) -> None:
        self._setup_shape(8, 4096)

    def benchmark_decode_m8(self) -> None:
        self.kernel.bias_gelu_quantize_fp8_static_bf16(
            self.input, self.bias, self.scale, out=self.out
        )

    def verify_decode_m8(self) -> torch.Tensor:
        return self._reference()

    def setup_small_m16(self) -> None:
        self._setup_shape(16, 4096)

    def benchmark_small_m16(self) -> None:
        self.kernel.bias_gelu_quantize_fp8_static_bf16(
            self.input, self.bias, self.scale, out=self.out
        )

    def verify_small_m16(self) -> torch.Tensor:
        return self._reference()

    def setup_prefill_m64(self) -> None:
        self._setup_shape(64, 4096)

    def benchmark_prefill_m64(self) -> None:
        self.kernel.bias_gelu_quantize_fp8_static_bf16(
            self.input, self.bias, self.scale, out=self.out
        )

    def verify_prefill_m64(self) -> torch.Tensor:
        return self._reference()

    def setup_prefill_m128(self) -> None:
        self._setup_shape(128, 4096)

    def benchmark_prefill_m128(self) -> None:
        self.kernel.bias_gelu_quantize_fp8_static_bf16(
            self.input, self.bias, self.scale, out=self.out
        )

    def verify_prefill_m128(self) -> torch.Tensor:
        return self._reference()

    def setup_wide_n8192_m16(self) -> None:
        self._setup_shape(16, 8192)

    def benchmark_wide_n8192_m16(self) -> None:
        self.kernel.bias_gelu_quantize_fp8_static_bf16(
            self.input, self.bias, self.scale, out=self.out
        )

    def verify_wide_n8192_m16(self) -> torch.Tensor:
        return self._reference()

    def setup_wide_n8192_m128(self) -> None:
        self._setup_shape(128, 8192)

    def benchmark_wide_n8192_m128(self) -> None:
        self.kernel.bias_gelu_quantize_fp8_static_bf16(
            self.input, self.bias, self.scale, out=self.out
        )

    def verify_wide_n8192_m128(self) -> torch.Tensor:
        return self._reference()


def _register_bias_gelu_shapes() -> None:
    for label, m, n in QUANT_SHAPES:

        def setup(self, m=m, n=n) -> None:
            self._setup_shape(m, n)

        def benchmark(self) -> None:
            self.kernel.bias_gelu_quantize_fp8_static_bf16(
                self.input, self.bias, self.scale, out=self.out
            )

        def verify(self) -> torch.Tensor:
            return self._reference()

        setattr(BiasGeluFp8QuantizeBenchmark, f"setup_{label}", setup)
        setattr(BiasGeluFp8QuantizeBenchmark, f"benchmark_{label}", benchmark)
        setattr(BiasGeluFp8QuantizeBenchmark, f"verify_{label}", verify)


class GeluFp8QuantizeBenchmark(Benchmark):
    seed = 3

    def _setup_shape(self, m: int, n: int) -> None:
        self.input = torch.randn((m, n), device=self.device, dtype=torch.bfloat16)
        self.scale = torch.tensor([0.25], device=self.device, dtype=torch.float32)
        self.out = torch.empty_like(self.input, dtype=_fp8_dtype())

    def _reference(self) -> torch.Tensor:
        y = torch.nn.functional.gelu(self.input.float(), approximate="tanh")
        limit = _fp8_max()
        y = torch.clamp(y / self.scale.float(), -limit, limit)
        return y.to(_fp8_dtype())

    def setup_decode_m1(self) -> None:
        self._setup_shape(1, 4096)

    def benchmark_decode_m1(self) -> None:
        self.kernel.gelu_quantize_fp8_static_bf16(
            self.input, self.scale, out=self.out
        )

    def verify_decode_m1(self) -> torch.Tensor:
        return self._reference()

    def setup_decode_m8(self) -> None:
        self._setup_shape(8, 4096)

    def benchmark_decode_m8(self) -> None:
        self.kernel.gelu_quantize_fp8_static_bf16(
            self.input, self.scale, out=self.out
        )

    def verify_decode_m8(self) -> torch.Tensor:
        return self._reference()

    def setup_small_m16(self) -> None:
        self._setup_shape(16, 4096)

    def benchmark_small_m16(self) -> None:
        self.kernel.gelu_quantize_fp8_static_bf16(
            self.input, self.scale, out=self.out
        )

    def verify_small_m16(self) -> torch.Tensor:
        return self._reference()

    def setup_prefill_m64(self) -> None:
        self._setup_shape(64, 4096)

    def benchmark_prefill_m64(self) -> None:
        self.kernel.gelu_quantize_fp8_static_bf16(
            self.input, self.scale, out=self.out
        )

    def verify_prefill_m64(self) -> torch.Tensor:
        return self._reference()

    def setup_prefill_m128(self) -> None:
        self._setup_shape(128, 4096)

    def benchmark_prefill_m128(self) -> None:
        self.kernel.gelu_quantize_fp8_static_bf16(
            self.input, self.scale, out=self.out
        )

    def verify_prefill_m128(self) -> torch.Tensor:
        return self._reference()

    def setup_wide_n8192_m16(self) -> None:
        self._setup_shape(16, 8192)

    def benchmark_wide_n8192_m16(self) -> None:
        self.kernel.gelu_quantize_fp8_static_bf16(
            self.input, self.scale, out=self.out
        )

    def verify_wide_n8192_m16(self) -> torch.Tensor:
        return self._reference()

    def setup_wide_n8192_m128(self) -> None:
        self._setup_shape(128, 8192)

    def benchmark_wide_n8192_m128(self) -> None:
        self.kernel.gelu_quantize_fp8_static_bf16(
            self.input, self.scale, out=self.out
        )

    def verify_wide_n8192_m128(self) -> torch.Tensor:
        return self._reference()


def _register_gelu_shapes() -> None:
    for label, m, n in QUANT_SHAPES:

        def setup(self, m=m, n=n) -> None:
            self._setup_shape(m, n)

        def benchmark(self) -> None:
            self.kernel.gelu_quantize_fp8_static_bf16(
                self.input, self.scale, out=self.out
            )

        def verify(self) -> torch.Tensor:
            return self._reference()

        setattr(GeluFp8QuantizeBenchmark, f"setup_{label}", setup)
        setattr(GeluFp8QuantizeBenchmark, f"benchmark_{label}", benchmark)
        setattr(GeluFp8QuantizeBenchmark, f"verify_{label}", verify)


_register_bias_gelu_shapes()
_register_gelu_shapes()
