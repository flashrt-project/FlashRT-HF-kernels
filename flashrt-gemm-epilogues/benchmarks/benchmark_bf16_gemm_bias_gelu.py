import os

import torch

from kernels.benchmark import Benchmark


PUBLIC_BF16_GEMM_WORKLOADS = {
    "bias_decode_m1",
    "gelu_decode_m1",
    "gelu_prefill_m64",
}

BF16_LINEAR_SHAPES = [
    ("pi05_action_in", 10, 32, 1024),
    ("pi05_qkv", 10, 1024, 2560),
    ("pi05_o_proj", 10, 2048, 1024),
    ("pi05_action_out", 10, 1024, 32),
    ("decode_m1_1024", 1, 1024, 1024),
    ("decode_m1_qkv", 1, 1024, 2560),
    ("decode_m8_1024", 8, 1024, 1024),
    ("decode_m8_qkv", 8, 1024, 2560),
    ("decode_m10_1024", 10, 1024, 1024),
    ("decode_m10_qkv", 10, 1024, 2560),
    ("decode_m16_1024", 16, 1024, 1024),
    ("decode_m16_qkv", 16, 1024, 2560),
    ("vlm_m512_square", 512, 1152, 1152),
    ("vlm_m512_wide", 512, 1152, 4304),
    ("vla_m1024_square", 1024, 2048, 2048),
    ("vla_m1024_wide", 1024, 2048, 8192),
]


class Bf16LinearBenchmark(Benchmark):
    seed = 4

    def _setup_shape(self, m: int, k: int, n: int) -> None:
        self.x = torch.randn((m, k), device=self.device, dtype=torch.bfloat16)
        self.w = torch.randn((k, n), device=self.device, dtype=torch.bfloat16)
        self.bias = torch.randn((n,), device=self.device, dtype=torch.bfloat16)
        self.out = torch.empty((m, n), device=self.device, dtype=torch.bfloat16)

    def _linear_reference(self) -> torch.Tensor:
        return (self.x @ self.w).to(torch.bfloat16)

    def _linear_bias_reference(self) -> torch.Tensor:
        return torch.addmm(self.bias, self.x, self.w).to(torch.bfloat16)


def _register_bf16_linear_shapes() -> None:
    for label, m, k, n in BF16_LINEAR_SHAPES:

        def setup(self, m=m, k=k, n=n) -> None:
            self._setup_shape(m, k, n)

        def benchmark(self) -> None:
            self.kernel.bf16_linear_bf16(self.x, self.w, out=self.out)

        def reference(self) -> torch.Tensor:
            return self._linear_reference()

        def setup_bias(self, m=m, k=k, n=n) -> None:
            self._setup_shape(m, k, n)

        def benchmark_bias(self) -> None:
            self.kernel.bf16_linear_bias_bf16(self.x, self.w, self.bias, out=self.out)

        def reference_bias(self) -> torch.Tensor:
            return self._linear_bias_reference()

        setattr(Bf16LinearBenchmark, f"setup_linear_{label}", setup)
        setattr(Bf16LinearBenchmark, f"benchmark_linear_{label}", benchmark)
        setattr(Bf16LinearBenchmark, f"reference_linear_{label}", reference)
        setattr(Bf16LinearBenchmark, f"setup_linear_bias_{label}", setup_bias)
        setattr(Bf16LinearBenchmark, f"benchmark_linear_bias_{label}", benchmark_bias)
        setattr(Bf16LinearBenchmark, f"reference_linear_bias_{label}", reference_bias)


class Bf16GemmEpilogueBenchmark(Benchmark):
    seed = 0

    def _setup_shape(self, m: int, n: int, k: int) -> None:
        self.a = torch.randn((m, k), device=self.device, dtype=torch.bfloat16)
        self.b = torch.randn((k, n), device=self.device, dtype=torch.bfloat16)
        self.bias = torch.randn((n,), device=self.device, dtype=torch.bfloat16)
        self.out = torch.empty((m, n), device=self.device, dtype=torch.bfloat16)

    def _bias_reference(self) -> torch.Tensor:
        return torch.addmm(self.bias, self.a, self.b).to(torch.bfloat16)

    def _gelu_reference(self) -> torch.Tensor:
        return torch.nn.functional.gelu(torch.addmm(self.bias, self.a, self.b)).to(
            torch.bfloat16
        )

    def setup_bias_decode_m1(self) -> None:
        self._setup_shape(1, 4096, 4096)

    def benchmark_bias_decode_m1(self) -> None:
        self.kernel.bf16_gemm_bias(self.a, self.b, self.bias, out=self.out)

    def reference_bias_decode_m1(self) -> torch.Tensor:
        return self._bias_reference()

    def setup_gelu_decode_m1(self) -> None:
        self._setup_shape(1, 4096, 4096)

    def benchmark_gelu_decode_m1(self) -> None:
        self.kernel.bf16_gemm_bias_gelu(self.a, self.b, self.bias, out=self.out)

    def reference_gelu_decode_m1(self) -> torch.Tensor:
        return self._gelu_reference()

    def setup_bias_decode_m8(self) -> None:
        self._setup_shape(8, 4096, 4096)

    def benchmark_bias_decode_m8(self) -> None:
        self.kernel.bf16_gemm_bias(self.a, self.b, self.bias, out=self.out)

    def reference_bias_decode_m8(self) -> torch.Tensor:
        return self._bias_reference()

    def setup_gelu_decode_m8(self) -> None:
        self._setup_shape(8, 4096, 4096)

    def benchmark_gelu_decode_m8(self) -> None:
        self.kernel.bf16_gemm_bias_gelu(self.a, self.b, self.bias, out=self.out)

    def reference_gelu_decode_m8(self) -> torch.Tensor:
        return self._gelu_reference()

    def setup_bias_small_m16(self) -> None:
        self._setup_shape(16, 4096, 4096)

    def benchmark_bias_small_m16(self) -> None:
        self.kernel.bf16_gemm_bias(self.a, self.b, self.bias, out=self.out)

    def reference_bias_small_m16(self) -> torch.Tensor:
        return self._bias_reference()

    def setup_gelu_small_m16(self) -> None:
        self._setup_shape(16, 4096, 4096)

    def benchmark_gelu_small_m16(self) -> None:
        self.kernel.bf16_gemm_bias_gelu(self.a, self.b, self.bias, out=self.out)

    def reference_gelu_small_m16(self) -> torch.Tensor:
        return self._gelu_reference()

    def setup_bias_prefill_m64(self) -> None:
        self._setup_shape(64, 4096, 4096)

    def benchmark_bias_prefill_m64(self) -> None:
        self.kernel.bf16_gemm_bias(self.a, self.b, self.bias, out=self.out)

    def reference_bias_prefill_m64(self) -> torch.Tensor:
        return self._bias_reference()

    def setup_gelu_prefill_m64(self) -> None:
        self._setup_shape(64, 4096, 4096)

    def benchmark_gelu_prefill_m64(self) -> None:
        self.kernel.bf16_gemm_bias_gelu(self.a, self.b, self.bias, out=self.out)

    def reference_gelu_prefill_m64(self) -> torch.Tensor:
        return self._gelu_reference()

    def setup_bias_prefill_m128(self) -> None:
        self._setup_shape(128, 4096, 4096)

    def benchmark_bias_prefill_m128(self) -> None:
        self.kernel.bf16_gemm_bias(self.a, self.b, self.bias, out=self.out)

    def reference_bias_prefill_m128(self) -> torch.Tensor:
        return self._bias_reference()

    def setup_gelu_prefill_m128(self) -> None:
        self._setup_shape(128, 4096, 4096)

    def benchmark_gelu_prefill_m128(self) -> None:
        self.kernel.bf16_gemm_bias_gelu(self.a, self.b, self.bias, out=self.out)

    def reference_gelu_prefill_m128(self) -> torch.Tensor:
        return self._gelu_reference()

    def setup_bias_wide_n8192_m16(self) -> None:
        self._setup_shape(16, 8192, 4096)

    def benchmark_bias_wide_n8192_m16(self) -> None:
        self.kernel.bf16_gemm_bias(self.a, self.b, self.bias, out=self.out)

    def reference_bias_wide_n8192_m16(self) -> torch.Tensor:
        return self._bias_reference()

    def setup_gelu_wide_n8192_m16(self) -> None:
        self._setup_shape(16, 8192, 4096)

    def benchmark_gelu_wide_n8192_m16(self) -> None:
        self.kernel.bf16_gemm_bias_gelu(self.a, self.b, self.bias, out=self.out)

    def reference_gelu_wide_n8192_m16(self) -> torch.Tensor:
        return self._gelu_reference()

    def setup_bias_wide_k8192_m16(self) -> None:
        self._setup_shape(16, 4096, 8192)

    def benchmark_bias_wide_k8192_m16(self) -> None:
        self.kernel.bf16_gemm_bias(self.a, self.b, self.bias, out=self.out)

    def reference_bias_wide_k8192_m16(self) -> torch.Tensor:
        return self._bias_reference()

    def setup_gelu_wide_k8192_m16(self) -> None:
        self._setup_shape(16, 4096, 8192)

    def benchmark_gelu_wide_k8192_m16(self) -> None:
        self.kernel.bf16_gemm_bias_gelu(self.a, self.b, self.bias, out=self.out)

    def reference_gelu_wide_k8192_m16(self) -> torch.Tensor:
        return self._gelu_reference()


def _hide_diagnostic_workloads() -> None:
    if os.environ.get("FLASHRT_ENABLE_DIAGNOSTIC_BF16_GEMM") == "1":
        return
    for name in list(vars(Bf16GemmEpilogueBenchmark)):
        for prefix in ("setup_", "benchmark_", "reference_"):
            if name.startswith(prefix):
                workload = name.removeprefix(prefix)
                if workload not in PUBLIC_BF16_GEMM_WORKLOADS:
                    delattr(Bf16GemmEpilogueBenchmark, name)
                break


_hide_diagnostic_workloads()
_register_bf16_linear_shapes()
