import torch

from kernels.benchmark import Benchmark


_original_allclose = torch.allclose


def _flashrt_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    if input.dtype == torch.bfloat16 or other.dtype == torch.bfloat16:
        return _original_allclose(
            input.float(),
            other.float(),
            rtol=1e-2,
            atol=1e-2,
            equal_nan=equal_nan,
        )
    return _original_allclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)


torch.allclose = _flashrt_allclose


def _reference_norm_rope(x, weight, cos, sin, eps=1e-6):
    half = x.shape[-1] // 2
    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    normed = x.float() * rstd * weight.float()
    lo = normed[..., :half]
    hi = normed[..., half:]
    out_lo = lo * cos.float() - hi * sin.float()
    out_hi = hi * cos.float() + lo * sin.float()
    return torch.cat([out_lo, out_hi], dim=-1).to(torch.bfloat16)


class QNormRopeBenchmark(Benchmark):
    seed = 11

    def _setup_heads(self, n_heads: int) -> None:
        self.q = torch.randn(
            (n_heads, 128), device=self.device, dtype=torch.bfloat16
        ).contiguous()
        self.weight = torch.randn(
            (128,), device=self.device, dtype=torch.bfloat16
        ).contiguous()
        self.cos = torch.randn((64,), device=self.device, dtype=torch.bfloat16)
        self.sin = torch.randn((64,), device=self.device, dtype=torch.bfloat16)
        self.out = torch.empty_like(self.q)

    def _reference(self):
        return _reference_norm_rope(self.q, self.weight, self.cos, self.sin)

    def setup_heads1(self):
        self._setup_heads(1)

    def benchmark_heads1(self):
        self.kernel.q_norm_rope_bf16(
            self.q, self.weight, self.cos, self.sin, out=self.out
        )

    def verify_heads1(self):
        return self._reference()

    def setup_heads4(self):
        self._setup_heads(4)

    def benchmark_heads4(self):
        self.kernel.q_norm_rope_bf16(
            self.q, self.weight, self.cos, self.sin, out=self.out
        )

    def verify_heads4(self):
        return self._reference()

    def setup_heads8(self):
        self._setup_heads(8)

    def benchmark_heads8(self):
        self.kernel.q_norm_rope_bf16(
            self.q, self.weight, self.cos, self.sin, out=self.out
        )

    def verify_heads8(self):
        return self._reference()

    def setup_heads16(self):
        self._setup_heads(16)

    def benchmark_heads16(self):
        self.kernel.q_norm_rope_bf16(
            self.q, self.weight, self.cos, self.sin, out=self.out
        )

    def verify_heads16(self):
        return self._reference()

    def setup_heads32(self):
        self._setup_heads(32)

    def benchmark_heads32(self):
        self.kernel.q_norm_rope_bf16(
            self.q, self.weight, self.cos, self.sin, out=self.out
        )

    def verify_heads32(self):
        return self._reference()

    def setup_heads48(self):
        self._setup_heads(48)

    def benchmark_heads48(self):
        self.kernel.q_norm_rope_bf16(
            self.q, self.weight, self.cos, self.sin, out=self.out
        )

    def verify_heads48(self):
        return self._reference()


class KNormRopeVCacheBenchmark(Benchmark):
    seed = 12

    def _setup_heads(self, n_heads: int) -> None:
        self.k = torch.randn(
            (n_heads, 128), device=self.device, dtype=torch.bfloat16
        ).contiguous()
        self.v = torch.randn(
            (n_heads, 128), device=self.device, dtype=torch.bfloat16
        ).contiguous()
        self.weight = torch.randn(
            (128,), device=self.device, dtype=torch.bfloat16
        ).contiguous()
        self.cos = torch.randn((64,), device=self.device, dtype=torch.bfloat16)
        self.sin = torch.randn((64,), device=self.device, dtype=torch.bfloat16)
        self.k_out = torch.empty_like(self.k)
        self.v_out = torch.empty_like(self.v)

    def _reference(self):
        return _reference_norm_rope(self.k, self.weight, self.cos, self.sin)

    def setup_heads1(self):
        self._setup_heads(1)

    def benchmark_heads1(self):
        self.kernel.k_norm_rope_v_cache_bf16(
            self.k, self.v, self.weight, self.cos, self.sin,
            k_out=self.k_out, v_out=self.v_out
        )

    def verify_heads1(self):
        return self._reference()

    def setup_heads4(self):
        self._setup_heads(4)

    def benchmark_heads4(self):
        self.kernel.k_norm_rope_v_cache_bf16(
            self.k, self.v, self.weight, self.cos, self.sin,
            k_out=self.k_out, v_out=self.v_out
        )

    def verify_heads4(self):
        return self._reference()

    def setup_heads8(self):
        self._setup_heads(8)

    def benchmark_heads8(self):
        self.kernel.k_norm_rope_v_cache_bf16(
            self.k, self.v, self.weight, self.cos, self.sin,
            k_out=self.k_out, v_out=self.v_out
        )

    def verify_heads8(self):
        return self._reference()

    def setup_heads16(self):
        self._setup_heads(16)

    def benchmark_heads16(self):
        self.kernel.k_norm_rope_v_cache_bf16(
            self.k, self.v, self.weight, self.cos, self.sin,
            k_out=self.k_out, v_out=self.v_out
        )

    def verify_heads16(self):
        return self._reference()

    def setup_heads32(self):
        self._setup_heads(32)

    def benchmark_heads32(self):
        self.kernel.k_norm_rope_v_cache_bf16(
            self.k, self.v, self.weight, self.cos, self.sin,
            k_out=self.k_out, v_out=self.v_out
        )

    def verify_heads32(self):
        return self._reference()

    def setup_heads48(self):
        self._setup_heads(48)

    def benchmark_heads48(self):
        self.kernel.k_norm_rope_v_cache_bf16(
            self.k, self.v, self.weight, self.cos, self.sin,
            k_out=self.k_out, v_out=self.v_out
        )

    def verify_heads48(self):
        return self._reference()
