from __future__ import annotations

import csv
import json
import math
import os
import time
import types
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM
from sparse_kv._C import fused_gqa

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE   = "cuda"
CTX_LEN  = 64000

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE)
model.eval()

input_ids = torch.randint(1000, 32000, (1, CTX_LEN), device=DEVICE, dtype=torch.long)
new_token = torch.randint(1000, 32000, (1, 1),        device=DEVICE, dtype=torch.long)


def prefill(model, input_ids: torch.Tensor) -> DynamicCache:
    """Run dense prefill, return a frozen DynamicCache regardless of transformers version."""
    restore_attention(model)
    with torch.no_grad():
        try:
            # Modern path: pass empty DynamicCache → output is DynamicCache
            out = model(
                input_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
            )
        except Exception:
            # Legacy path: pass None → output is tuple-of-tuples
            # clone_cache() handles the conversion transparently
            out = model(
                input_ids,
                past_key_values=None,
                use_cache=True,
            )
    # clone_cache handles both DynamicCache and legacy tuple format
    return clone_cache(out.past_key_values)

def patch_attention(
    model,
    mode: str = "token-sparse",
    top_k: int = 32,
    block_size: int = 64,
    top_k_blocks: int = 8,
    sparse_layers: Optional[List[int]] = None,
) -> int:
    """
    Replace LlamaAttention.forward with a sparse decode kernel.

    Args:
        mode          : "token-sparse" | "block-sparse" | "dense"
        top_k         : tokens attended (token-sparse)
        block_size    : tokens per block (block-sparse)
        top_k_blocks  : blocks attended (block-sparse)
        sparse_layers : if given, only patch these layer indices;
                        all others remain dense.  Pass None to patch all.
                        Example: sparse_layers=list(range(16)) → first-16 hybrid.

    Returns patched layer count.
    """
    def sparse_forward(
        self, hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        B, S, _ = hidden_states.shape
        H   = self.config.num_attention_heads
        D   = self.head_dim
        Hkv = self.config.num_key_value_heads

        Q = self.q_proj(hidden_states).view(B, S, H,   D).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, S, Hkv, D).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, S, Hkv, D).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(V, position_ids)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if past_key_value is not None:
            # Upgrade legacy tuple cache to DynamicCache on first decode step
            if not isinstance(past_key_value, DynamicCache):
                past_key_value = DynamicCache.from_legacy_cache(past_key_value)
            
            if not hasattr(self, "layer_idx") or self.layer_idx is None:
                raise ValueError("self.layer_idx missing — check your transformers version.")
            
            cache_kwargs = {"sin": sin, "cos": cos}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            K, V = past_key_value.update(K, V, self.layer_idx, cache_kwargs)

        K = repeat_kv(K, H // Hkv)   # [B, H, ctx_len, D]
        V = repeat_kv(V, H // Hkv)

        kv_len = K.shape[2]

        if S == 1:
            # ── decode step: use sparse kernel ────────────────────────────
            if mode == "token-sparse" and kv_len >= top_k:
                attn_out = kv_evict_quant_forward(
                    Q.contiguous().to(torch.float16),
                    K.contiguous().to(torch.float16),
                    V.contiguous().to(torch.float16),
                    top_k, False,
                ).to(Q.dtype)

            elif mode == "block-sparse":
                attn_out = sparse_attention_forward(
                    Q.contiguous().to(torch.float16),
                    K.contiguous().to(torch.float16),
                    V.contiguous().to(torch.float16),
                    block_size, top_k_blocks,
                ).to(Q.dtype)

            elif mode == "v3-gqa":
                # 1. Squeeze the sequence dimension: [B, Hq, 1, D] -> [B, Hq, D]
                q_in = Q.squeeze(2).contiguous().to(torch.float16)
                
                # 2. Grab the raw GQA cache: [B, Hkv, N, D] -> permute to [B, N, Hkv, D]
                k_in = past_key_value.key_cache[self.layer_idx].permute(0, 2, 1, 3).contiguous().to(torch.float16)
                v_in = past_key_value.value_cache[self.layer_idx].permute(0, 2, 1, 3).contiguous().to(torch.float16)
                
                scale = 1.0 / math.sqrt(D)
                
                # 3. Execute your custom kernel! Returns [B, Hq, D]
                attn_out = fused_gqa(q_in, k_in, v_in, scale)
                
                # 4. Unsqueeze sequence dim back to [B, Hq, 1, D] so HuggingFace can flatten it
                attn_out = attn_out.unsqueeze(2).to(Q.dtype)

            else:
                # Fallback to dense PyTorch SDPA
                attn_out = F.scaled_dot_product_attention(
                    Q, K, V, attn_mask=None, is_causal=False
                )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, H * D)
        attn_out = self.o_proj(attn_out)

        present_key_value = past_key_value if use_cache else None
        return attn_out, None, present_key_value

    patched = 0
    for i, layer in enumerate(model.model.layers):
        if sparse_layers is not None and i not in sparse_layers:
            continue
        attn = layer.self_attn
        if not hasattr(attn, "_orig_forward"):
            attn._orig_forward = attn.forward
        attn.forward = types.MethodType(sparse_forward, attn)
        patched += 1

    print(f"✅ Patched {patched} layers | mode={mode} | "
          f"sparse_layers={'all' if sparse_layers is None else sparse_layers}")
    return patched

def restore_attention(model) -> int:
    """
    Restore all patched attention layers to their original forward.
    Fix [P2-7]: deletes _orig_forward after restore so the function
    is idempotent — calling it twice does not double-restore.
    """
    restored = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_orig_forward"):
            attn.forward = attn._orig_forward
            del attn._orig_forward
            restored += 1
    if restored:
        print(f"✅ Restored {restored} attention layers")
    return restored


def clone_cache(past_kv) -> DynamicCache:
    """
    Deep-copy a KV cache regardless of whether it is a legacy tuple-of-tuples
    (older transformers) or a DynamicCache instance (v4.36+).

    The warning "passing past_key_values as a tuple is deprecated" means your
    transformers version returns tuples from model() — DynamicCache.from_legacy_cache()
    converts it, then we clone each tensor so the original is never mutated.
    """
    if isinstance(past_kv, DynamicCache):
        # Modern path: DynamicCache with .key_cache / .value_cache lists
        new_cache = DynamicCache()
        for layer_idx in range(len(past_kv)):
            new_cache.update(
                past_kv.key_cache[layer_idx].clone(),
                past_kv.value_cache[layer_idx].clone(),
                layer_idx,
            )
        return new_cache
    else:
        # Legacy path: tuple of (key, value) tensors per layer
        # Convert → DynamicCache so the rest of the code is uniform
        new_cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past_kv):
            new_cache.update(
                k.clone(),
                v.clone(),
                layer_idx,
            )
        return new_cache


past_kv = prefill(model, input_ids)

# ── warmup (NOT profiled) ─────────────────────────────────────────────────────
for _ in range(3):
    with torch.no_grad():
        restore_attention(model)
        model(new_token, past_key_values=clone_cache(past_kv), use_cache=True)

torch.cuda.synchronize()

# ── DENSE decode — profiled ───────────────────────────────────────────────────
torch.cuda.cudart().cudaProfilerStart()

nvtx.range_push("dense_decode")
with torch.no_grad():
    restore_attention(model)
    model(new_token, past_key_values=clone_cache(past_kv), use_cache=True)
torch.cuda.synchronize()
nvtx.range_pop()

# ── V3 decode — profiled ──────────────────────────────────────────────────────
nvtx.range_push("v3_full_model_decode")
with torch.no_grad():
    patch_attention(model, mode="v3-gqa")
    model(new_token, past_key_values=clone_cache(past_kv), use_cache=True)
    restore_attention(model)
torch.cuda.synchronize()
nvtx.range_pop()

# ── raw v3 kernel only — profiled ─────────────────────────────────────────────
cfg   = model.config
Hq    = cfg.num_attention_heads
Hkv   = cfg.num_key_value_heads
D     = cfg.hidden_size // Hq
scale = 1.0 / (D ** 0.5)

K0 = past_kv.key_cache[0].permute(0,2,1,3).contiguous()
V0 = past_kv.value_cache[0].permute(0,2,1,3).contiguous()
Q0 = torch.randn(1, Hq, D, dtype=torch.float16, device=DEVICE)

nvtx.range_push("v3_kernel_only")
with torch.no_grad():
    fused_gqa(Q0, K0, V0, scale)
torch.cuda.synchronize()
nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()

print("Profile written to results/v3_profile.nsys-rep")
print("Open with: nsys-ui results/v3_profile.nsys-rep")