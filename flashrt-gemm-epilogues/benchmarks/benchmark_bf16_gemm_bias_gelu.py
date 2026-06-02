import torch

from kernels.benchmark import Benchmark


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
