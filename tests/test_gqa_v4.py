"""Correctness gate for the v4 decode kernel (block per KV head).

v4 changes the work decomposition, not the maths: it must stay numerically
equivalent to v3 and to PyTorch SDPA. A performance rewrite that quietly
changes the output is worse than no rewrite, so this runs before any v4
number is believed.
"""
import pytest
import torch
import torch.nn.functional as F

import sparse_kv._C as _C

HQ, HKV, D = 32, 8, 128
SCALE = 1.0 / (D ** 0.5)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(_C, "fused_gqa_v4"),
    reason="needs CUDA and a build containing fused_gqa_v4")


def _ref(Q, K, V):
    """SDPA reference; expands KV heads when torch is too old for enable_gqa."""
    Qs = Q.unsqueeze(2)
    Ks = K.transpose(1, 2).contiguous()
    Vs = V.transpose(1, 2).contiguous()
    try:
        return F.scaled_dot_product_attention(
            Qs, Ks, Vs, scale=SCALE, enable_gqa=True).squeeze(2)
    except TypeError:
        B, N = K.size(0), K.size(1)
        n = HQ // HKV
        Ke = Ks[:, :, None].expand(B, HKV, n, N, D).reshape(B, HQ, N, D).contiguous()
        Ve = Vs[:, :, None].expand(B, HKV, n, N, D).reshape(B, HQ, N, D).contiguous()
        return F.scaled_dot_product_attention(Qs, Ke, Ve, scale=SCALE).squeeze(2)


def _tensors(B, N, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, HQ, D, device="cuda", dtype=torch.float16),
            torch.randn(B, N, HKV, D, device="cuda", dtype=torch.float16),
            torch.randn(B, N, HKV, D, device="cuda", dtype=torch.float16))


@pytest.mark.parametrize("N", [1, 63, 64, 65, 128, 512, 4096, 16384])
def test_v4_matches_sdpa(N):
    """Includes N around the 64-token tile boundary, where partial tiles live."""
    Q, K, V = _tensors(1, N)
    out = _C.fused_gqa_v4(Q, K, V, SCALE)
    ref = _ref(Q, K, V)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 5e-3, "max abs err {} at N={}".format(err, N)


@pytest.mark.parametrize("B", [1, 2, 4])
def test_v4_batched(B):
    Q, K, V = _tensors(B, 2048)
    out = _C.fused_gqa_v4(Q, K, V, SCALE)
    ref = _ref(Q, K, V)
    assert (out.float() - ref.float()).abs().max().item() < 5e-3


def test_v4_agrees_with_v3():
    """The two kernels must be interchangeable, not merely both 'close enough'."""
    Q, K, V = _tensors(1, 8192)
    a = _C.fused_gqa_v4(Q, K, V, SCALE).float()
    b = _C.fused_gqa(Q, K, V, SCALE).float()
    assert (a - b).abs().max().item() < 5e-3


def test_v4_no_nan():
    Q, K, V = _tensors(1, 4096)
    out = _C.fused_gqa_v4(Q, K, V, SCALE)
    assert torch.isfinite(out).all()
