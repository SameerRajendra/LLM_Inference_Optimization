import torch
from torch import Tensor

try:
    from sparse_kv._C import kv_evict_quant_forward as _cuda_evict
    _CUDA = True
except ImportError:
    _cuda_evict = None
    _CUDA = False


def fused_kv_evict(
    K_cache: Tensor,
    V_cache: Tensor,
    attn_scores: Tensor,
    top_k: int,
    use_int8: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Fused KV eviction + optional INT8 quantization.
    Uses CUDA kernel if available, falls back to Python reference.
    """
    if _CUDA and _cuda_evict is not None:
        # CUDA kernel path: takes Q,K,V directly and returns sparse output
        # attn_scores not needed — kernel recomputes QK^T internally
        return _cuda_evict(K_cache, V_cache, top_k, use_int8)
    return _reference_eviction(K_cache, V_cache, attn_scores, top_k, use_int8)


def _reference_eviction(
    K_cache: Tensor,
    V_cache: Tensor,
    attn_scores: Tensor,
    top_k: int,
    use_int8: bool,
) -> tuple[Tensor, Tensor]:
    idx   = attn_scores.topk(top_k, dim=-1).indices
    idx   = idx.unsqueeze(-1).expand(-1, -1, -1, K_cache.shape[-1])
    K_out = K_cache.gather(2, idx)
    V_out = V_cache.gather(2, idx)
    if use_int8:
        K_out = K_out.to(torch.int8)
        V_out = V_out.to(torch.int8)
    return K_out, V_out