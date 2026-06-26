import torch
import pytest
from sparse_kv.attention import _reference_sparse_attention
from sparse_kv.eviction  import _reference_eviction

B, H, S, D = 1, 8, 512, 64


def test_sparse_attention_shape():
    Q = torch.randn(B, H, 16, D)
    K, V = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    out = _reference_sparse_attention(Q, K, V, top_k=64, sink_tokens=4)
    assert out.shape == (B, H, 16, D)


def test_sparse_attention_no_nan():
    Q = torch.randn(B, H, 16, D)
    K, V = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    assert not torch.isnan(_reference_sparse_attention(Q, K, V, 64, 4)).any()


def test_eviction_shape():
    K, V   = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    K_out, V_out = _reference_eviction(K, V, torch.rand(B, H, S), 128, False)
    assert K_out.shape == (B, H, 128, D)


def test_eviction_int8_dtype():
    K, V   = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    K_out, V_out = _reference_eviction(K, V, torch.rand(B, H, S), 64, True)
    assert K_out.dtype == torch.int8
    assert V_out.dtype == torch.int8