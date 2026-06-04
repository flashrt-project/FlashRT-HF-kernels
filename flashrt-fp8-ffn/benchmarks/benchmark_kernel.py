import torch

from kernels.benchmark import Benchmark


def _quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)


def _dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float()


def _compiler_disable(fn):
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(fn)
    return torch._dynamo.disable(fn)


def _gelu_quantize_fp8_boundary(
    hidden: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    hidden = torch.nn.functional.gelu(
        hidden.float() + bias.float(), approximate="tanh"
    )
    return _quantize_fp8(hidden, scale)


def _bf16_bias_add_boundary(out: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return (out.float() + bias.float()).to(torch.bfloat16)


_stable_gelu_quantize_fp8 = _compiler_disable(_gelu_quantize_fp8_boundary)
_stable_bf16_bias_add = _compiler_disable(_bf16_bias_add_boundary)


class FP8GeluMlpBenchmark(Benchmark):
    seed = 17

    def _setup_shape(self, M: int, K: int, H: int, N: int) -> None:
        self.M, self.K, self.H, self.N = M, K, H, N
        self.x_scale = torch.tensor([0.05], device=self.device, dtype=torch.float32)
        self.up_scale = torch.tensor([0.04], device=self.device, dtype=torch.float32)
        self.hidden_scale = torch.tensor([0.25], device=self.device, dtype=torch.float32)
        self.down_scale = torch.tensor([0.04], device=self.device, dtype=torch.float32)
        self.x = _quantize_fp8(
            torch.randn((M, K), device=self.device, dtype=torch.bfloat16),
            self.x_scale,
        )
        self.up_w = _quantize_fp8(
            torch.randn((H, K), device=self.device, dtype=torch.bfloat16),
            self.up_scale,
        )
        self.down_w = _quantize_fp8(
            torch.randn((N, H), device=self.device, dtype=torch.bfloat16),
            self.down_scale,
        )
        self.up_b = torch.randn((H,), device=self.device, dtype=torch.bfloat16)
        self.down_b = torch.randn((N,), device=self.device, dtype=torch.bfloat16)
        self.hidden = torch.empty((M, H), device=self.device, dtype=torch.bfloat16)
        self.hidden_fp8 = torch.empty((M, H), device=self.device, dtype=torch.float8_e4m3fn)
        self.out = torch.empty((M, N), device=self.device, dtype=torch.bfloat16)

    def _reference(self) -> torch.Tensor:
        hidden = (
            _dequant_fp8(self.x, self.x_scale)
            @ _dequant_fp8(self.up_w, self.up_scale).T
        ).to(torch.bfloat16)
        hidden_fp8 = _stable_gelu_quantize_fp8(
            hidden, self.up_b, self.hidden_scale
        )
        out = (
            _dequant_fp8(hidden_fp8, self.hidden_scale)
            @ _dequant_fp8(self.down_w, self.down_scale).T
        ).to(torch.bfloat16)
        return _stable_bf16_bias_add(out, self.down_b)

    def setup_smoke_mlp(self) -> None:
        self._setup_shape(16, 128, 256, 128)

    def benchmark_smoke_mlp(self) -> None:
        self.kernel.fp8_gelu_mlp_bf16(
            self.x,
            self.up_w,
            self.up_b,
            self.down_w,
            self.down_b,
            self.x_scale,
            self.up_scale,
            self.hidden_scale,
            self.down_scale,
            hidden_bf16=self.hidden,
            hidden_fp8=self.hidden_fp8,
            out=self.out,
        )

    def verify_smoke_mlp(self) -> torch.Tensor:
        return self._reference()
