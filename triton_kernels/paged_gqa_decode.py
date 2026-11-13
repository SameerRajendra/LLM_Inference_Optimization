"""Split-KV GQA decode over a paged KV pool with ragged seq_lens (WP-1).

Same tensor-core scheme as gqa_decode.py (the G query heads of one KV head form the M dimension
of tl.dot), but K/V are read through a block table:
    q            [B, Hq, D] fp16
    k/v_cache    [num_blocks, block_size, Hkv, D] fp16, or uint8 e4m3 bits
    k/v_scale    [num_blocks, block_size // 64, Hkv] fp32 (FP8 only; see paged_kv.py)
    block_table  [B, max_blocks_per_seq] int32, logical block -> physical block
    seq_lens     [B] int32
Splits are sized from the longest sequence; a split that starts past seq_lens[b] runs no tiles
and writes an empty partial (m = -inf, l = 0), which the combine kernel weights by exactly 0.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .gqa_decode import KV_PAGE_TOKENS, LOG2E, MAX_SPLITS

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def _prune(configs, named_args, **kwargs):
    """A tile must sit inside one physical block (one block-table lookup per tile) and, for FP8,
    inside one 64-token scale page (one scale per tile)."""
    bs = kwargs["BLOCK_SIZE"]
    ok = [c for c in configs if bs % c.kwargs["BLOCK_N"] == 0
          and (not kwargs["FP8"] or KV_PAGE_TOKENS % c.kwargs["BLOCK_N"] == 0)]
    if INTERPRET:  # no GPU to benchmark on: take the widest legal tile
        ok = [max(ok, key=lambda c: c.kwargs["BLOCK_N"])]
    return ok


CONFIGS = [triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=s)
           for bn in (16, 32, 64, 128) for w in (4, 8) for s in (2, 3)]


@triton.autotune(configs=CONFIGS, key=["tokens_bucket", "FP8", "BLOCK_SIZE", "D", "G"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _paged_split_kernel(
    Q, K, V, KS, VS, BT, SL, O, M, L,
    bt_stride, split_len, num_splits, qk_scale,
    tokens_bucket,  # autotune key only: batch * max_seq_len, rounded to a power of 2
    Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    PAGE: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, FP8: tl.constexpr,
):
    """One program = (batch, kv head, KV split)."""
    pid, split = tl.program_id(0), tl.program_id(1)
    b = (pid // Hkv).to(tl.int64)
    h = pid % Hkv
    Hq = Hkv * G
    rows = tl.arange(0, BLOCK_H)
    cols = tl.arange(0, D)
    offs = tl.arange(0, BLOCK_N)
    row_ok = rows < G

    q = tl.load(Q + ((b * Hq + h * G + rows) * D)[:, None] + cols[None, :],
                mask=row_ok[:, None], other=0.0)

    seq_len = tl.load(SL + b)
    start = split * split_len
    end = tl.minimum(start + split_len, seq_len)  # <= start for a split past the sequence
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)

    for n0 in range(start, end, BLOCK_N):
        # split_len is a multiple of BLOCK_SIZE and BLOCK_N divides it, so the tile is one block.
        blk = tl.load(BT + b * bt_stride + n0 // BLOCK_SIZE).to(tl.int64)
        in_blk = n0 % BLOCK_SIZE
        in_range = n0 + offs < end
        tile = (((blk * BLOCK_SIZE + in_blk + offs) * Hkv + h) * D)[:, None] + cols[None, :]
        k = tl.load(K + tile, mask=in_range[:, None], other=0)
        v = tl.load(V + tile, mask=in_range[:, None], other=0)
        if FP8:
            k = k.to(tl.float8e4nv, bitcast=True)
            v = v.to(tl.float8e4nv, bitcast=True)
            sc = (blk * (BLOCK_SIZE // PAGE) + in_blk // PAGE) * Hkv + h
            ks = tl.load(KS + sc)
            vs = tl.load(VS + sc)
        else:
            ks = 1.0
            vs = 1.0

        qk = tl.dot(q, tl.trans(k.to(tl.float16))) * (qk_scale * ks)
        qk = tl.where(in_range[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16)) * vs
        m_i = m_new

    row = (b * Hq + h * G + rows) * num_splits + split
    tl.store(O + (row * D)[:, None] + cols[None, :], acc, mask=row_ok[:, None])
    tl.store(M + row, m_i, mask=row_ok)
    tl.store(L + row, l_i, mask=row_ok)


@triton.jit
def _paged_combine_kernel(O, M, L, Out, num_splits, D: tl.constexpr, BLOCK_S: tl.constexpr):
    """Exact log-sum-exp merge. Empty splits (m = -inf) get weight 0; a row whose splits are
    all empty (seq_len 0) produces 0 rather than NaN."""
    row = tl.program_id(0).to(tl.int64)
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, D)
    ok = s < num_splits
    m = tl.load(M + row * num_splits + s, mask=ok, other=float("-inf"))
    denom = tl.load(L + row * num_splits + s, mask=ok, other=0.0)
    o = tl.load(O + ((row * num_splits + s) * D)[:, None] + d[None, :],
                mask=ok[:, None], other=0.0)
    m_max = tl.max(m, 0)
    m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
    w = tl.exp2(m - m_max)
    total = tl.sum(denom * w, 0)
    out = tl.sum(o * w[:, None], 0) / tl.where(total > 0, total, 1.0)
    tl.store(Out + row * D + d, out.to(tl.float16))


def paged_gqa_decode(q, k_cache, v_cache, block_table, seq_lens, scale=None,
                     k_scale=None, v_scale=None, max_seq_len=None, num_splits=None):
    """Decode attention for one new token per sequence over a paged pool.

    Passing k_scale/v_scale selects the FP8 path. max_seq_len (>= seq_lens.max()) sizes the
    split grid; pass it to avoid a device -> host sync. num_splits overrides the heuristic.
    """
    B, Hq, D = q.shape
    num_blocks, block_size, Hkv, Dk = k_cache.shape
    fp8 = k_scale is not None
    assert D == Dk and D in (64, 128) and Hq % Hkv == 0, "needs D in {64, 128}, Hq % Hkv == 0"
    assert q.dtype == torch.float16 and k_cache.dtype == (torch.uint8 if fp8 else torch.float16)
    assert v_cache.shape == k_cache.shape and k_cache.is_contiguous() and v_cache.is_contiguous()
    assert block_size >= 16 and block_size & (block_size - 1) == 0, "block_size: power of 2 >= 16"
    assert block_table.dtype == torch.int32 and block_table.shape[0] == B
    assert seq_lens.dtype == torch.int32 and seq_lens.shape == (B,)
    assert block_table.stride(1) == 1, "block_table rows must be contiguous"
    if fp8:
        assert block_size % KV_PAGE_TOKENS == 0, "FP8 pool needs block_size % 64 == 0"
        pages = num_blocks * (block_size // KV_PAGE_TOKENS)
        for s in (k_scale, v_scale):
            assert s.dtype == torch.float32 and s.is_contiguous() and s.numel() == pages * Hkv
    G = Hq // Hkv
    q = q.contiguous()
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item()) if B > 0 else 0
    max_seq_len = max(int(max_seq_len), 1)
    max_blocks = triton.cdiv(max_seq_len, block_size)

    if num_splits is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count if q.is_cuda else 1
        num_splits = min(triton.cdiv(2 * sms, B * Hkv), max_blocks)
    num_splits = max(1, min(num_splits, max_blocks, MAX_SPLITS))
    # Block-aligned splits: every tile maps to exactly one physical block.
    split_len = triton.cdiv(max_blocks, num_splits) * block_size
    num_splits = triton.cdiv(max_seq_len, split_len)

    f32 = dict(dtype=torch.float32, device=q.device)
    O = torch.empty(B, Hq, num_splits, D, **f32)
    M = torch.empty(B, Hq, num_splits, **f32)
    L = torch.empty(B, Hq, num_splits, **f32)
    out = torch.empty_like(q)
    if B == 0:
        return out

    scale = D ** -0.5 if scale is None else scale
    _paged_split_kernel[(B * Hkv, num_splits)](
        q, k_cache, v_cache, k_scale, v_scale, block_table, seq_lens, O, M, L,
        block_table.stride(0), split_len, num_splits, scale * LOG2E,
        triton.next_power_of_2(B * max_seq_len),
        Hkv=Hkv, G=G, D=D, BLOCK_SIZE=block_size, PAGE=KV_PAGE_TOKENS,
        BLOCK_H=max(16, triton.next_power_of_2(G)), FP8=fp8)
    _paged_combine_kernel[(B * Hq,)](O, M, L, out, num_splits, D=D,
                                     BLOCK_S=triton.next_power_of_2(num_splits))
    return out
