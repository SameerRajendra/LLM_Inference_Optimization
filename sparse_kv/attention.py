import torch
from torch import Tensor


def sparse_attention(
    Q: Tensor, K: Tensor, V: Tensor,
    top_k: int = 64,
    sink_tokens: int = 4,
) -> Tensor:
    """Sparse attention — uses CUDA kernel if built, else PyTorch reference."""
    try:
        from sparse_kv._C import kv_evict_quant_forward
        orig_dtype = Q.dtype
        out = kv_evict_quant_forward(
            Q.to(torch.float16),
            K.to(torch.float16),
            V.to(torch.float16),
            top_k, False
        )
        return out.to(orig_dtype)
    except ImportError:
        return _reference_sparse_attention(Q, K, V, top_k, sink_tokens)


def _reference_sparse_attention(Q, K, V, top_k, sink_tokens):
    scale  = Q.shape[-1] ** -0.5
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) * scale

    sink_mask = torch.zeros_like(scores, dtype=torch.bool)
    sink_mask[..., :sink_tokens] = True

    topk_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, scores.topk(top_k, dim=-1).indices, True
    )

    scores = scores.masked_fill(~(sink_mask | topk_mask), float("-inf"))
    attn   = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", attn, V)