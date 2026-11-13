import pytest
import torch

from sparse_kv.attention import sparse_attention

try:
    from sparse_kv._C import kv_evict_quant_forward
except ImportError:
    kv_evict_quant_forward = None


@pytest.mark.skipif(
    not torch.cuda.is_available() or kv_evict_quant_forward is None,
    reason="needs CUDA and built sparse_kv._C",
)
def test_kv_evict_quant_correctness():
    B, H, N, D, TOP_K = 1, 32, 4096, 128, 32
    torch.manual_seed(42)

    Q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    cuda_out = kv_evict_quant_forward(Q, K, V, TOP_K, False)
    ref_out = sparse_attention(Q, K, V, TOP_K, sink_tokens=0)

    diff = (cuda_out - ref_out).abs().max().item()
    assert diff < 0.15, "Kernel diverged: max diff = {:.5f}".format(diff)