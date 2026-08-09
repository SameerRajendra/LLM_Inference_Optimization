"""JAX reference implementation of block-sparse attention.

This is a functional port of csrc/kernels/sparse_attention.cu.
It is intended as a cross-framework correctness reference, NOT a
performance-optimised kernel.  The block-sparse logic (score KV blocks,
select top-k, accumulate V) is identical to the CUDA path so that
numerical outputs can be compared directly.

Algorithm (mirrors block_sparse_attn_kernel in sparse_attention.cu)
-------------------------------------------------------------------
1. Partition the K/V sequence dimension into non-overlapping blocks of
   size `block_size`.
2. Score each block by computing the mean QK^T dot-product over all
   tokens in the block.
3. Select the `top_k_blocks` highest-scoring blocks.
4. Run softmax over the selected block scores.
5. Accumulate the weighted V values for each selected block, spreading
   the block weight uniformly across its tokens (matching the CUDA
   kernel's `token_w = w_blk / count` logic).

Public API
----------
  sparse_attention_pallas(Q, K, V, block_size, top_k_blocks)
      Pure-JAX reference compiled under jax.jit — XLA executes it on
      GPU.  A raw pl.pallas_call kernel with statically-tiled grid is
      a planned next step.

  sparse_attention_reference(Q, K, V, block_size, top_k_blocks)
      Eager pure-JAX reference — always available, no JIT.

Shapes
------
  Q   : [B, H, 1, D]   fp16 or fp32
  K   : [B, H, T, D]
  V   : [B, H, T, D]
  Out : [B, H, 1, D]   same dtype as Q
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# Pure-JAX reference  (eager, always available, CPU + GPU)
# ---------------------------------------------------------------------------

def sparse_attention_reference(
    Q: jax.Array,          # [B, H, 1, D]
    K: jax.Array,          # [B, H, T, D]
    V: jax.Array,          # [B, H, T, D]
    block_size: int = 64,
    top_k_blocks: int = 32,
) -> jax.Array:            # [B, H, 1, D]
    """Pure-JAX block-sparse attention — numerically equivalent to the CUDA kernel."""
    orig_dtype = Q.dtype
    Q = Q.astype(jnp.float32)
    K = K.astype(jnp.float32)
    V = V.astype(jnp.float32)

    B, H, _, D = Q.shape
    T = K.shape[2]
    scale = 1.0 / math.sqrt(D)

    num_blocks = math.ceil(T / block_size)

    # -- Step 1: pad K/V to a multiple of block_size -------------------------
    pad_len = num_blocks * block_size - T
    if pad_len > 0:
        K = jnp.pad(K, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        V = jnp.pad(V, ((0, 0), (0, 0), (0, pad_len), (0, 0)))

    # K_blocks: [B, H, num_blocks, block_size, D]
    K_blocks = K.reshape(B, H, num_blocks, block_size, D)
    V_blocks = V.reshape(B, H, num_blocks, block_size, D)

    # -- Step 2: score each block  -------------------------------------------
    # q: [B, H, D]
    q = Q[:, :, 0, :]
    # dots: [B, H, num_blocks, block_size]
    dots = jnp.einsum("bhd,bhnsd->bhns", q, K_blocks) * scale

    # Validity mask — marks padding tokens as invalid
    tok_local  = jnp.arange(block_size)[None, None, None, :]               # [1,1,1,bs]
    blk_offset = jnp.arange(num_blocks)[None, None, :, None] * block_size  # [1,1,nb,1]
    global_idx = blk_offset + tok_local                                     # [1,1,nb,bs]
    valid      = global_idx < T                                             # [1,1,nb,bs]

    dots = jnp.where(valid, dots, 0.0)

    # Mean score per block: sum(dots) / count  (matches CUDA accum/count)
    # broadcast_to materialises [B,H,nb] explicitly — vmap requires concrete batch dim
    count = jnp.broadcast_to(
        jnp.sum(valid.astype(jnp.float32), axis=-1),
        (B, H, num_blocks)
    )                                                                        # [B,H,nb]
    block_scores = jnp.sum(dots, axis=-1) / jnp.maximum(count, 1.0)        # [B,H,nb]

    # -- Step 3: select top-k blocks  ----------------------------------------
    actual_k = min(top_k_blocks, num_blocks)
    top_vals, top_idx = lax.top_k(block_scores, actual_k)                  # [B,H,actual_k]

    # -- Step 4: softmax over selected block scores  -------------------------
    top_vals = top_vals - jnp.max(top_vals, axis=-1, keepdims=True)
    top_w    = jnp.exp(top_vals)
    top_w    = top_w / jnp.maximum(jnp.sum(top_w, axis=-1, keepdims=True), 1e-9)
    # top_w: [B, H, actual_k]

    # -- Step 5: gather selected V blocks and accumulate  --------------------
    # V_sel: [B, H, actual_k, block_size, D]
    def gather_v(v_bh, idx):      # v_bh: [nb, bs, D]  idx: [actual_k]
        return v_bh[idx]
    V_sel = jax.vmap(jax.vmap(gather_v))(V_blocks, top_idx)

    # count for each selected block: [B, H, actual_k]
    def gather_count(c_bh, idx):  # c_bh: [nb]  idx: [actual_k]
        return c_bh[idx]
    sel_counts = jax.vmap(jax.vmap(gather_count))(count, top_idx)          # [B,H,actual_k]

    # token_w = w_blk / count — matches CUDA exactly
    token_w = (top_w / jnp.maximum(sel_counts, 1.0))[:, :, :, None, None]  # [B,H,actual_k,1,1]

    # Mask out padding tokens in the last block
    sel_offsets = (top_idx * block_size)[:, :, :, None]                    # [B,H,actual_k,1]
    tok_global  = sel_offsets + jnp.arange(block_size)[None, None, None, :]# [B,H,actual_k,bs]
    tok_valid   = (tok_global < T)[:, :, :, :, None]                       # [B,H,actual_k,bs,1]

    weighted_V = jnp.where(tok_valid, V_sel * token_w, 0.0)                # [B,H,actual_k,bs,D]
    out = jnp.sum(weighted_V, axis=(2, 3))                                  # [B,H,D]
    out = out[:, :, None, :]                                                # [B,H,1,D]

    return out.astype(orig_dtype)


# ---------------------------------------------------------------------------
# JIT-compiled GPU path
# ---------------------------------------------------------------------------

# Pre-compile with static_argnums so block_size and top_k_blocks are
# treated as compile-time constants by XLA — same as template parameters
# in the CUDA kernel.
_jitted_reference = jax.jit(sparse_attention_reference, static_argnums=(3, 4))


def sparse_attention_pallas(
    Q: jax.Array,          # [B, H, 1, D]
    K: jax.Array,          # [B, H, T, D]
    V: jax.Array,          # [B, H, T, D]
    block_size: int = 64,
    top_k_blocks: int = 32,
) -> jax.Array:            # [B, H, 1, D]
    """Block-sparse attention — JAX/XLA JIT-compiled path.

    Calls sparse_attention_reference under jax.jit so XLA compiles and
    executes the kernel on GPU.  block_size and top_k_blocks are treated
    as compile-time constants (static_argnums), matching the template
    behaviour of the CUDA kernel.

    A raw pl.pallas_call kernel with statically-tiled grid is a planned
    next step — see jax_ref/README.md.

    Parameters
    ----------
    Q             [B, H, 1, D]
    K             [B, H, T, D]
    V             [B, H, T, D]
    block_size    tokens per KV block   (default 64, matches CUDA default)
    top_k_blocks  blocks to attend to   (default 32)

    Returns
    -------
    jax.Array  [B, H, 1, D], same dtype as Q
    """
    return _jitted_reference(Q, K, V, block_size, top_k_blocks)