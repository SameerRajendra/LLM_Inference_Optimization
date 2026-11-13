from typing import Tuple

import torch
from torch import Tensor


def fused_kv_evict(
    K_cache: Tensor,
    V_cache: Tensor,
    attn_scores: Tensor,
    top_k: int,
    use_int8: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Evict KV entries outside top_k by attention score, with optional INT8 cast.

    Note: for fused QK^T + top-k + weighted V attention output directly in CUDA,
    use `sparse_kv.attention.sparse_attention` (`kv_evict_quant_forward`).
    This helper returns the pruned `(K_out, V_out)` cache tensors.
    """
    return _reference_eviction(K_cache, V_cache, attn_scores, top_k, use_int8)


def _reference_eviction(
    K_cache: Tensor,
    V_cache: Tensor,
    attn_scores: Tensor,
    top_k: int,
    use_int8: bool,
) -> Tuple[Tensor, Tensor]:
    k = min(top_k, K_cache.shape[2])
    idx = attn_scores.topk(k, dim=-1).indices
    idx = idx.unsqueeze(-1).expand(-1, -1, -1, K_cache.shape[-1])
    K_out = K_cache.gather(2, idx)
    V_out = V_cache.gather(2, idx)
    if use_int8:
        K_out = K_out.to(torch.int8)
        V_out = V_out.to(torch.int8)
    return K_out, V_out