import torch

import sparse_kv
from sparse_kv.attention import sparse_attention
from sparse_kv.eviction import fused_kv_evict


def test_package_exports_safe_without_cuda():
    assert hasattr(sparse_kv, "CUDA_AVAILABLE")
    assert hasattr(sparse_kv, "fused_kv_evict")
    assert hasattr(sparse_kv, "sparse_attention")


def test_fused_kv_evict_returns_pruned_kv_fp16():
    B, H, N, D = 2, 4, 128, 64
    K = torch.randn(B, H, N, D, dtype=torch.float16)
    V = torch.randn(B, H, N, D, dtype=torch.float16)
    scores = torch.rand(B, H, N)

    k_out, v_out = fused_kv_evict(K, V, scores, top_k=32, use_int8=False)
    assert k_out.shape == (B, H, 32, D)
    assert v_out.shape == (B, H, 32, D)
    assert k_out.dtype == torch.float16


def test_fused_kv_evict_clamps_topk_above_seq_len():
    B, H, N, D = 1, 2, 16, 32
    K = torch.randn(B, H, N, D, dtype=torch.float16)
    V = torch.randn(B, H, N, D, dtype=torch.float16)
    scores = torch.rand(B, H, N)

    k_out, v_out = fused_kv_evict(K, V, scores, top_k=64, use_int8=True)
    assert k_out.shape == (B, H, N, D)
    assert k_out.dtype == torch.int8
    assert v_out.dtype == torch.int8


def test_sparse_attention_cpu_fallback():
    B, H, N, D = 1, 4, 64, 32
    Q = torch.randn(B, H, 1, D, dtype=torch.float32)
    K = torch.randn(B, H, N, D, dtype=torch.float32)
    V = torch.randn(B, H, N, D, dtype=torch.float32)
    out = sparse_attention(Q, K, V, top_k=16, sink_tokens=4)
    assert out.shape == (B, H, 1, D)
    assert torch.isfinite(out).all()
