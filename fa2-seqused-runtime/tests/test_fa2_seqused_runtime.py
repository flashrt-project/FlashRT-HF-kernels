from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from fa2_seqused_runtime import (
    allocate_outputs,
    allocate_workspace,
    forward,
    forward_seqused_static,
    forward_static,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    hq, hkv = q.shape[2], k.shape[2]
    if hq != hkv:
        repeat = hq // hkv
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
    attn_mask = None
    if causal:
        sq, sk = q.shape[1], k.shape[1]
        q_idx = torch.arange(sq, device=q.device).view(sq, 1)
        k_idx = torch.arange(sk, device=q.device).view(1, sk)
        attn_mask = k_idx <= q_idx + sk - sq
    return F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        attn_mask=attn_mask,
        is_causal=False,
    ).permute(0, 2, 1, 3)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float() - expected.float()).abs()
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    if actual.dtype == torch.float16:
        assert diff.max().item() <= 8e-3
        assert diff.mean().item() <= 4e-4
        assert cosine.item() >= 0.9999
    else:
        assert diff.max().item() <= 6.5e-2
        assert diff.mean().item() <= 3e-3
        assert cosine.item() >= 0.999


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 8, 8),
        (1, 17, 63, 16, 4),
        (2, 64, 129, 12, 4),
        (1, 257, 511, 8, 2),
    ],
)
def test_noncausal_full_matrix(dtype, head_dim, shape):
    batch, sq, sk, hq, hkv = shape
    q = torch.randn(batch, sq, hq, head_dim, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(batch, sk, hkv, head_dim, device="cuda", dtype=dtype) * 0.5
    v = torch.randn_like(k)
    actual = forward(q, k, v, use_split_kv=False)
    expected = _reference(q, k, v)
    _assert_close(actual, expected)


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("seqlen", [1, 17, 128, 257, 1024])
def test_causal_bf16(head_dim, seqlen):
    q = torch.randn(1, seqlen, 8, head_dim, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(1, seqlen, 2, head_dim, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k)
    actual = forward(q, k, v, causal=True, use_split_kv=False)
    _assert_close(actual, _reference(q, k, v, causal=True))


def test_aligned_padded_strides_bf16():
    def padded(shape):
        storage = torch.randn(*shape[:-1], shape[-1] + 8, device="cuda", dtype=torch.bfloat16)
        return storage[..., : shape[-1]]

    q = padded((1, 49, 8, 128))
    k = padded((1, 257, 2, 128))
    v = padded((1, 257, 2, 128))
    out = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
    lse = torch.empty((1, 8, 49), device="cuda", dtype=torch.float32)
    forward_static(q, k, v, out=out, softmax_lse=lse)
    _assert_close(out, _reference(q, k, v))


def test_device_seqused_per_batch():
    q = torch.randn(2, 17, 8, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(2, 513, 2, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k)
    seqused = torch.tensor([127, 513], device="cuda", dtype=torch.int32)
    out, lse = allocate_outputs(q)
    forward_seqused_static(q, k, v, seqused, out=out, softmax_lse=lse)
    refs = []
    for batch, used in enumerate((127, 513)):
        refs.append(_reference(q[batch : batch + 1], k[batch : batch + 1, :used], v[batch : batch + 1, :used]))
    _assert_close(out, torch.cat(refs, dim=0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [96, 128, 256])
def test_split_kv_noncausal(dtype, head_dim):
    q = torch.randn(1, 1, 8, head_dim, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(1, 4096, 2, head_dim, device="cuda", dtype=dtype) * 0.5
    v = torch.randn_like(k)
    workspace = allocate_workspace(q, k)
    assert workspace is not None and workspace.num_splits > 1
    out, lse = allocate_outputs(q)
    forward_static(q, k, v, out=out, softmax_lse=lse, workspace=workspace)
    _assert_close(out, _reference(q, k, v))


def test_split_kv_seqused():
    q = torch.randn(1, 1, 8, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(1, 4096, 2, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k)
    workspace = allocate_workspace(q, k)
    assert workspace is not None and workspace.num_splits > 1
    out, lse = allocate_outputs(q)
    used = torch.tensor([3073], device="cuda", dtype=torch.int32)
    forward_seqused_static(
        q, k, v, used, out=out, softmax_lse=lse, workspace=workspace
    )
    _assert_close(out, _reference(q, k[:, :3073], v[:, :3073]))


@pytest.mark.parametrize("head_dim", [128, 256])
def test_split_kv_causal(head_dim):
    q = torch.randn(1, 257, 8, head_dim, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(1, 4096, 2, head_dim, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k)
    workspace = allocate_workspace(q, k)
    assert workspace is not None and workspace.num_splits > 1
    out, lse = allocate_outputs(q)
    forward_static(q, k, v, out=out, softmax_lse=lse, workspace=workspace, causal=True)
    _assert_close(out, _reference(q, k, v, causal=True))


def test_cuda_graph_static_and_device_length_update():
    q = torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k)
    used = torch.tensor([128], device="cuda", dtype=torch.int32)
    out, lse = allocate_outputs(q)
    for _ in range(3):
        forward_seqused_static(q, k, v, used, out=out, softmax_lse=lse)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward_seqused_static(q, k, v, used, out=out, softmax_lse=lse)
    graph.replay()
    _assert_close(out, _reference(q, k[:, :128], v[:, :128]))
    used.fill_(255)
    graph.replay()
    _assert_close(out, _reference(q, k[:, :255], v[:, :255]))


def test_torch_compile_trace():
    q = torch.randn(1, 16, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 64, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out, lse = allocate_outputs(q)

    def call(q_, k_, v_, out_, lse_):
        return forward_static(q_, k_, v_, out=out_, softmax_lse=lse_)

    compiled = torch.compile(call, fullgraph=True)
    actual = compiled(q, k, v, out, lse)
    _assert_close(actual, _reference(q, k, v))


@pytest.mark.parametrize("bad_dim", [32, 80, 192])
def test_rejects_unbuilt_head_dim(bad_dim):
    q = torch.randn(1, 4, 4, bad_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 4, 4, bad_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    lse = torch.empty((1, 4, 4), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="head_dim"):
        forward_static(q, k, v, out=out, softmax_lse=lse)


def test_rejects_fp16_causal_and_split_hdim64():
    q = torch.randn(1, 4, 4, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 4, 4, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    out, lse = allocate_outputs(q)
    with pytest.raises(RuntimeError, match="causal v1 supports bf16"):
        forward_static(q, k, v, out=out, softmax_lse=lse, causal=True)

    q64 = torch.randn(1, 1, 4, 64, device="cuda", dtype=torch.bfloat16)
    k64 = torch.randn(1, 4096, 4, 64, device="cuda", dtype=torch.bfloat16)
    assert allocate_workspace(q64, k64) is None


def test_rejects_misaligned_head_stride():
    storage = torch.randn(1, 8, 4, 129, device="cuda", dtype=torch.bfloat16)
    q = storage[..., :128]
    k = storage[..., :128]
    v = storage[..., :128]
    out = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
    lse = torch.empty((1, 4, 8), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="preserve 16-byte alignment"):
        forward_static(q, k, v, out=out, softmax_lse=lse)
