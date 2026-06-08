import torch

from kernels.benchmark import Benchmark


def _quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float() / scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)


def _dequant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.float() * scale.float()


class FP8SwiGLUMlpBenchmark(Benchmark):
    seed = 17

    def _setup_shape(self, M: int, K: int, H: int, N: int) -> None:
        self.M, self.K, self.H, self.N = M, K, H, N
        self.x_scale = torch.tensor([0.05], device=self.device, dtype=torch.float32)
        self.gate_up_scale = torch.tensor([0.04], device=self.device, dtype=torch.float32)
        self.hidden_scale = torch.tensor([0.25], device=self.device, dtype=torch.float32)
        self.down_scale = torch.tensor([0.04], device=self.device, dtype=torch.float32)
        self.x = _quantize_fp8(
            torch.randn((M, K), device=self.device, dtype=torch.bfloat16),
            self.x_scale,
        )
        self.gate_up_w = _quantize_fp8(
            torch.randn((2 * H, K), device=self.device, dtype=torch.bfloat16),
            self.gate_up_scale,
        )
        self.down_w = _quantize_fp8(
            torch.randn((N, H), device=self.device, dtype=torch.bfloat16),
            self.down_scale,
        )
        self.gate_up_bf16 = torch.empty((M, 2 * H), device=self.device, dtype=torch.bfloat16)
        self.hidden_fp8 = torch.empty((M, H), device=self.device, dtype=torch.float8_e4m3fn)
        self.out = torch.empty((M, N), device=self.device, dtype=torch.bfloat16)

    def _reference(self) -> torch.Tensor:
        gate_up = (
            _dequant_fp8(self.x, self.x_scale)
            @ _dequant_fp8(self.gate_up_w, self.gate_up_scale).T
        ).to(torch.bfloat16)
        gate, up = gate_up.float().chunk(2, dim=1)
        hidden_fp8 = _quantize_fp8(torch.nn.functional.silu(gate) * up, self.hidden_scale)
        return (
            _dequant_fp8(hidden_fp8, self.hidden_scale)
            @ _dequant_fp8(self.down_w, self.down_scale).T
        ).to(torch.bfloat16)

    def setup_smoke_mlp(self) -> None:
        self._setup_shape(10, 1024, 4096, 1024)

    def benchmark_smoke_mlp(self) -> None:
        self.kernel.fp8_swiglu_mlp_bf16(
            self.x,
            self.gate_up_w,
            self.down_w,
            self.x_scale,
            self.gate_up_scale,
            self.hidden_scale,
            self.down_scale,
            gate_up_bf16=self.gate_up_bf16,
            hidden_fp8=self.hidden_fp8,
            out=self.out,
        )

    def verify_smoke_mlp(self) -> torch.Tensor:
        return self._reference()
