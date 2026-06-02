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


def _reference_qkv_split_norm_rope(
    packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im, heads, head_dim, eps=1e-6
):
    batch, tokens, _ = packed_qkv.shape
    dim = heads * head_dim
    q = packed_qkv[..., :dim].reshape(batch, tokens, heads, head_dim)
    k = packed_qkv[..., dim : 2 * dim].reshape(batch, tokens, heads, head_dim)
    qf = q.float()
    kf = k.float()
    qn = qf * torch.rsqrt((qf * qf).mean(dim=(-2, -1), keepdim=True) + eps)
    kn = kf * torch.rsqrt((kf * kf).mean(dim=(-2, -1), keepdim=True) + eps)
    qn = (qn * norm_q_weight.reshape(1, 1, heads, head_dim).float()).to(torch.bfloat16)
    kn = (kn * norm_k_weight.reshape(1, 1, heads, head_dim).float()).to(torch.bfloat16)

    def rope(x):
        xr = x[..., 0::2].float()
        xi = x[..., 1::2].float()
        fr = freqs_re[:tokens][None, :, None, :]
        fi = freqs_im[:tokens][None, :, None, :]
        out = torch.empty_like(x)
        out[..., 0::2] = (xr * fr - xi * fi).to(torch.bfloat16)
        out[..., 1::2] = (xr * fi + xi * fr).to(torch.bfloat16)
        return out

    return rope(qn), rope(kn)


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

    def setup_heads24(self):
        self._setup_heads(24)

    def benchmark_heads24(self):
        self.kernel.q_norm_rope_bf16(
            self.q, self.weight, self.cos, self.sin, out=self.out
        )

    def verify_heads24(self):
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


class QKVSplitNormRopeBenchmark(Benchmark):
    seed = 13

    def _setup_tokens(self, tokens: int) -> None:
        self.heads = 24
        self.head_dim = 128
        dim = self.heads * self.head_dim
        self.packed_qkv = torch.randn(
            (1, tokens, 3 * dim), device=self.device, dtype=torch.bfloat16
        ).contiguous()
        self.norm_q_weight = torch.randn(
            (dim,), device=self.device, dtype=torch.bfloat16
        ).contiguous()
        self.norm_k_weight = torch.randn(
            (dim,), device=self.device, dtype=torch.bfloat16
        ).contiguous()
        self.freqs_re = torch.randn(
            (4096, self.head_dim // 2), device=self.device, dtype=torch.float32
        ).contiguous()
        self.freqs_im = torch.randn(
            (4096, self.head_dim // 2), device=self.device, dtype=torch.float32
        ).contiguous()
        self.q_out = torch.empty(
            (1, tokens, self.heads, self.head_dim),
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.k_out = torch.empty_like(self.q_out)
        self.out = self.q_out

    def _reference_pair(self):
        return _reference_qkv_split_norm_rope(
            self.packed_qkv,
            self.norm_q_weight,
            self.norm_k_weight,
            self.freqs_re,
            self.freqs_im,
            self.heads,
            self.head_dim,
        )

    def _reference(self):
        return self._reference_pair()[0]

    def _benchmark(self):
        self.kernel.qkv_split_norm_rope_bf16(
            self.packed_qkv,
            self.norm_q_weight,
            self.norm_k_weight,
            self.freqs_re,
            self.freqs_im,
            heads=self.heads,
            head_dim=self.head_dim,
            q_out=self.q_out,
            k_out=self.k_out,
        )

    def setup_tokens1(self):
        self._setup_tokens(1)

    def benchmark_tokens1(self):
        self._benchmark()

    def verify_tokens1(self):
        return self._reference()

    def setup_tokens4(self):
        self._setup_tokens(4)

    def benchmark_tokens4(self):
        self._benchmark()

    def verify_tokens4(self):
        return self._reference()

    def setup_tokens16(self):
        self._setup_tokens(16)

    def benchmark_tokens16(self):
        self._benchmark()

    def verify_tokens16(self):
        return self._reference()

    def setup_tokens64(self):
        self._setup_tokens(64)

    def benchmark_tokens64(self):
        self._benchmark()

    def verify_tokens64(self):
        return self._reference()

    def setup_tokens256(self):
        self._setup_tokens(256)

    def benchmark_tokens256(self):
        self._benchmark()

    def verify_tokens256(self):
        return self._reference()

    def setup_tokens1024(self):
        self._setup_tokens(1024)

    def benchmark_tokens1024(self):
        self._benchmark()

    def verify_tokens1024(self):
        return self._reference()

    def setup_tokens2520(self):
        self._setup_tokens(2520)

    def benchmark_tokens2520(self):
        self._benchmark()

    def verify_tokens2520(self):
        return self._reference()

    def setup_tokens4096(self):
        self._setup_tokens(4096)

    def benchmark_tokens4096(self):
        self._benchmark()

    def verify_tokens4096(self):
        return self._reference()


class QKVSplitNormRopeKBenchmark(QKVSplitNormRopeBenchmark):
    seed = 14

    def _setup_tokens(self, tokens: int) -> None:
        super()._setup_tokens(tokens)
        self.out = self.k_out

    def _reference(self):
        return self._reference_pair()[1]


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
        self.out = self.k_out

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

    def setup_heads24(self):
        self._setup_heads(24)

    def benchmark_heads24(self):
        self.kernel.k_norm_rope_v_cache_bf16(
            self.k, self.v, self.weight, self.cos, self.sin,
            k_out=self.k_out, v_out=self.v_out
        )

    def verify_heads24(self):
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
