"""Quest-style query-aware page-sparse decode over the paged KV pool (WP-4).

Two stages per decode step, all on device with no host sync (CUDA-graph capturable when
max_seq_len is passed):

  Stage 1  _quest_score_kernel  reads only the per-page key bounds k_min/k_max [num_blocks,
           Hkv, D] (paged_kv.allocate_page_minmax) and writes, per logical page p,
               score[b, h, p] = max_{g in group} sum_d max(q_{g,d} * k_min[phys, h, d],
                                                         q_{g,d} * k_max[phys, h, d])
           an upper bound on every q.k in the page. Because k_min <= k_max, the max picks k_max
           where q > 0 and k_min where q < 0, so the bound is two tensor-core dots:
               max(q+, 0) @ k_max^T + min(q, 0) @ k_min^T   (fp16 products, fp32 accumulate).
           Page 0 (attention sink) and the last `recent_pages` pages of each sequence get
           +inf, pages at or past seq_lens[b] get -inf.
  select   torch.topk over the scores -> selected_pages [B, Hkv, top_k] int32 logical page ids,
           sorted ascending, -1 for unused slots (sequences shorter than top_k pages).
  Stage 2  _quest_split_kernel  split-KV flash decode (same tile math as paged_gqa_decode) that
           walks ONLY the selected pages, so K/V HBM traffic is proportional to
           top_k / total pages. fp16 or FP8 (e4m3 + per-64-token scales) pools, ragged seq_lens.
           The partials are merged by paged_gqa_decode's exact log-sum-exp combine kernel.

With top_k_pages >= the number of pages, every valid page is selected and the result equals
dense paged_gqa_decode up to fp32 summation order.

Page = one physical block of the pool (block_size tokens). The metadata costs 2 * D * 2 bytes
per (page, kv head) per step, i.e. 1 / block_size of the page's fp16 K+V bytes (2 / block_size
for an FP8 pool), and it is read for EVERY page, so small pages trade selection granularity for
metadata traffic (at 16-token pages and a 12.5% budget it is half the selected K/V bytes).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .gqa_decode import KV_PAGE_TOKENS, LOG2E, MAX_SPLITS
from .paged_gqa_decode import CONFIGS, _paged_combine_kernel, _prune

SCORE_BLOCK_P = 32  # pages scored per program


@triton.jit
def _quest_score_kernel(
    Q, KMIN, KMAX, BT, SL, S,
    bt_stride, max_pages, recent_pages,
    Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_P: tl.constexpr,
):
    """One program = (batch, kv head, BLOCK_P logical pages)."""
    pid, pb = tl.program_id(0), tl.program_id(1)
    b = (pid // Hkv).to(tl.int64)
    h = pid % Hkv
    Hq = Hkv * G
    rows = tl.arange(0, BLOCK_H)
    cols = tl.arange(0, D)
    p = pb * BLOCK_P + tl.arange(0, BLOCK_P)
    row_ok = rows < G

    q = tl.load(Q + ((b * Hq + h * G + rows) * D)[:, None] + cols[None, :],
                mask=row_ok[:, None], other=0.0)
    q_pos = tl.where(q > 0, q, 0.0).to(tl.float16)
    q_neg = tl.where(q < 0, q, 0.0).to(tl.float16)

    seq_len = tl.load(SL + b)
    n_pages = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    p_ok = p < n_pages
    blk = tl.load(BT + b * bt_stride + p, mask=p_ok, other=0).to(tl.int64)
    meta = ((blk * Hkv + h) * D)[:, None] + cols[None, :]
    k_min = tl.load(KMIN + meta, mask=p_ok[:, None], other=0.0)
    k_max = tl.load(KMAX + meta, mask=p_ok[:, None], other=0.0)

    s = tl.dot(q_pos, tl.trans(k_max)) + tl.dot(q_neg, tl.trans(k_min))  # [BLOCK_H, BLOCK_P]
    s = tl.where(row_ok[:, None], s, float("-inf"))
    score = tl.max(s, 0)
    forced = (p == 0) | (p >= n_pages - recent_pages)
    score = tl.where(forced, float("inf"), score)
    score = tl.where(p_ok, score, float("-inf"))
    tl.store(S + (b * Hkv + h) * max_pages + p, score, mask=p < max_pages)


@triton.autotune(configs=CONFIGS, key=["sel_bucket", "FP8", "BLOCK_SIZE", "D", "G"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _quest_split_kernel(
    Q, K, V, KS, VS, BT, SL, SEL, O, M, L,
    bt_stride, top_k, pages_per_split, num_splits, qk_scale,
    sel_bucket,  # autotune key only: B * top_k * block_size, rounded to a power of 2
    Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    PAGE: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, FP8: tl.constexpr,
):
    """One program = (batch, kv head, chunk of the selected-page list). Unselected pages are
    never addressed, so their K/V bytes never leave HBM."""
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
    n_pages = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    sel_base = (b * Hkv + h) * top_k
    i0 = split * pages_per_split
    i1 = tl.minimum(i0 + pages_per_split, top_k)
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)

    for i in range(i0, i1):
        lp = tl.load(SEL + sel_base + i)
        ok = (lp >= 0) & (lp < n_pages)
        blk = tl.load(BT + b * bt_stride + lp, mask=ok, other=0).to(tl.int64)
        # Valid tokens in this page; 0 for an unused slot, so its loop runs no tiles.
        ntok = tl.where(ok, tl.minimum(seq_len - lp * BLOCK_SIZE, BLOCK_SIZE), 0)
        for n0 in range(0, ntok, BLOCK_N):
            in_range = n0 + offs < ntok
            tile = (((blk * BLOCK_SIZE + n0 + offs) * Hkv + h) * D)[:, None] + cols[None, :]
            k = tl.load(K + tile, mask=in_range[:, None], other=0)
            v = tl.load(V + tile, mask=in_range[:, None], other=0)
            if FP8:
                k = k.to(tl.float8e4nv, bitcast=True)
                v = v.to(tl.float8e4nv, bitcast=True)
                sc = (blk * (BLOCK_SIZE // PAGE) + n0 // PAGE) * Hkv + h
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


# ----------------------------------------------------------------------------- host wrappers

def _max_pages(seq_lens, block_size, max_seq_len):
    if max_seq_len is None:  # device -> host sync; pass max_seq_len to avoid it
        max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
    return max(triton.cdiv(max(int(max_seq_len), 1), block_size), 1)


def pages_for_budget(budget, max_seq_len, block_size, recent_pages=1):
    """top_k_pages for a fractional page budget of the longest sequence, never fewer than the
    always-kept sink + recent pages."""
    total = triton.cdiv(max(int(max_seq_len), 1), block_size)
    return min(total, max(int(round(budget * total)), 1 + recent_pages))


def quest_page_scores(q, k_min, k_max, block_table, seq_lens, block_size, recent_pages=1,
                      max_seq_len=None):
    """Stage 1: upper-bound page scores [B, Hkv, max_pages] fp32 (+inf forced, -inf invalid)."""
    B, Hq, D = q.shape
    num_blocks, Hkv, Dk = k_min.shape
    assert D == Dk and D in (64, 128) and Hq % Hkv == 0
    assert q.dtype == torch.float16 and k_min.dtype == torch.float16
    assert k_max.shape == k_min.shape and k_max.dtype == torch.float16
    assert k_min.is_contiguous() and k_max.is_contiguous()
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert seq_lens.dtype == torch.int32 and seq_lens.shape == (B,)
    assert recent_pages >= 0
    G = Hq // Hkv
    max_pages = _max_pages(seq_lens, block_size, max_seq_len)
    scores = torch.empty(B, Hkv, max_pages, dtype=torch.float32, device=q.device)
    if B == 0:
        return scores
    _quest_score_kernel[(B * Hkv, triton.cdiv(max_pages, SCORE_BLOCK_P))](
        q.contiguous(), k_min, k_max, block_table, seq_lens, scores,
        block_table.stride(0), max_pages, int(recent_pages),
        Hkv=Hkv, G=G, D=D, BLOCK_SIZE=block_size,
        BLOCK_H=max(16, triton.next_power_of_2(G)), BLOCK_P=SCORE_BLOCK_P)
    return scores


def quest_select_pages(scores, top_k_pages):
    """Top-k logical pages per (b, kv head): [B, Hkv, min(top_k_pages, max_pages)] int32,
    ascending, with -1 in slots a short sequence cannot fill. Forced pages score +inf, so they
    are always kept as long as top_k_pages >= 1 + recent_pages."""
    max_pages = scores.shape[-1]
    k = max(1, min(int(top_k_pages), max_pages))
    vals, idx = torch.topk(scores, k, dim=-1, sorted=False)
    idx = torch.where(vals > float("-inf"), idx, torch.full_like(idx, max_pages))
    idx, _ = torch.sort(idx, dim=-1)  # ascending page order; unused slots sort last
    return torch.where(idx < max_pages, idx, torch.full_like(idx, -1)).to(torch.int32)


def paged_sparse_decode(q, k_cache, v_cache, block_table, seq_lens, selected_pages, scale=None,
                        k_scale=None, v_scale=None, num_splits=None):
    """Stage 2: decode attention over only the logical pages in selected_pages [B, Hkv, top_k]
    (int32, -1 = unused). Passing k_scale/v_scale selects the FP8 path."""
    B, Hq, D = q.shape
    num_blocks, block_size, Hkv, Dk = k_cache.shape
    fp8 = k_scale is not None
    assert D == Dk and D in (64, 128) and Hq % Hkv == 0, "needs D in {64, 128}, Hq % Hkv == 0"
    assert q.dtype == torch.float16 and k_cache.dtype == (torch.uint8 if fp8 else torch.float16)
    assert v_cache.shape == k_cache.shape and k_cache.is_contiguous() and v_cache.is_contiguous()
    assert block_size >= 16 and block_size & (block_size - 1) == 0, "block_size: power of 2 >= 16"
    assert block_table.dtype == torch.int32 and block_table.shape[0] == B
    assert block_table.stride(1) == 1, "block_table rows must be contiguous"
    assert seq_lens.dtype == torch.int32 and seq_lens.shape == (B,)
    assert selected_pages.dtype == torch.int32 and selected_pages.dim() == 3
    assert selected_pages.shape[:2] == (B, Hkv)
    if fp8:
        assert block_size % KV_PAGE_TOKENS == 0, "FP8 pool needs block_size % 64 == 0"
        pages = num_blocks * (block_size // KV_PAGE_TOKENS)
        for s in (k_scale, v_scale):
            assert s.dtype == torch.float32 and s.is_contiguous() and s.numel() == pages * Hkv
    G = Hq // Hkv
    top_k = selected_pages.shape[2]
    selected_pages = selected_pages.contiguous()
    q = q.contiguous()

    if num_splits is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count if q.is_cuda else 1
        num_splits = triton.cdiv(2 * sms, max(B * Hkv, 1))
    num_splits = max(1, min(num_splits, top_k, MAX_SPLITS))
    pages_per_split = triton.cdiv(top_k, num_splits)
    num_splits = triton.cdiv(top_k, pages_per_split)

    f32 = dict(dtype=torch.float32, device=q.device)
    O = torch.empty(B, Hq, num_splits, D, **f32)
    M = torch.empty(B, Hq, num_splits, **f32)
    L = torch.empty(B, Hq, num_splits, **f32)
    out = torch.empty_like(q)
    if B == 0 or top_k == 0:
        return out.zero_()

    scale = D ** -0.5 if scale is None else scale
    _quest_split_kernel[(B * Hkv, num_splits)](
        q, k_cache, v_cache, k_scale, v_scale, block_table, seq_lens, selected_pages, O, M, L,
        block_table.stride(0), top_k, pages_per_split, num_splits, scale * LOG2E,
        triton.next_power_of_2(B * top_k * block_size),
        Hkv=Hkv, G=G, D=D, BLOCK_SIZE=block_size, PAGE=KV_PAGE_TOKENS,
        BLOCK_H=max(16, triton.next_power_of_2(G)), FP8=fp8)
    _paged_combine_kernel[(B * Hq,)](O, M, L, out, num_splits, D=D,
                                     BLOCK_S=triton.next_power_of_2(num_splits))
    return out


def quest_sparse_decode(q, k_cache, v_cache, block_table, seq_lens, k_min, k_max, top_k_pages,
                        recent_pages=1, scale=None, k_scale=None, v_scale=None,
                        max_seq_len=None, num_splits=None, return_selected=False):
    """Score pages, keep the top_k_pages (+ sink page 0 + last recent_pages), attend over them.

    max_seq_len (>= seq_lens.max(), fixed per captured graph) avoids the only host sync, so the
    whole call can be captured in a CUDA graph after a warm-up with the same shapes.
    Returns out [B, Hq, D] fp16, or (out, selected_pages) when return_selected.
    """
    block_size = k_cache.shape[1]
    assert int(top_k_pages) >= 1 + int(recent_pages), \
        "top_k_pages must cover the sink page and the recent_pages window"
    scores = quest_page_scores(q, k_min, k_max, block_table, seq_lens, block_size,
                               recent_pages, max_seq_len)
    selected = quest_select_pages(scores, top_k_pages)
    out = paged_sparse_decode(q, k_cache, v_cache, block_table, seq_lens, selected, scale=scale,
                              k_scale=k_scale, v_scale=v_scale, num_splits=num_splits)
    return (out, selected) if return_selected else out


# ------------------------------------------------------------------------- bandwidth accounting

def quest_hbm_bytes(selected_pages, seq_lens, block_size, num_q_heads, head_dim, fp8=False,
                    max_pages=None):
    """Bytes the sparse path must move from/to HBM for one decode step, counted from what the
    kernels address (each byte once, no cache reuse), next to the dense paged kernel's bytes.
    Host-synchronising: call it outside timed regions."""
    B, Hkv, top_k = selected_pages.shape
    sl = seq_lens.long().view(B, 1, 1)
    elt = 1 if fp8 else 2
    sel = selected_pages.long()
    ntok = torch.where(sel >= 0, torch.clamp(sl - sel * block_size, 0, block_size),
                       torch.zeros_like(sel))
    sel_tokens = int(ntok.sum().item())
    n_sel_pages = int((ntok > 0).sum().item())
    n_pages = int(((seq_lens.long() + block_size - 1) // block_size).sum().item()) * Hkv
    dense_tokens = int(seq_lens.long().sum().item()) * Hkv
    if max_pages is None:
        max_pages = top_k

    scale_pages = triton.cdiv(block_size, KV_PAGE_TOKENS)
    kv_tok = 2 * head_dim * elt                                # K + V, one token, one kv head
    q_out = 2 * B * num_q_heads * head_dim * 2                 # q read + out write (fp16)
    sparse_kv = sel_tokens * kv_tok + n_sel_pages * (4 + (8 * scale_pages if fp8 else 0))
    metadata = n_pages * (2 * head_dim * 2 + 4)                # k_min + k_max + block_table
    selection = B * Hkv * (max_pages * 4 * 2 + top_k * 4 * 2)  # scores w+r, selected w+r
    dense_kv = dense_tokens * kv_tok + n_pages * (4 + (8 * scale_pages if fp8 else 0))
    sparse_total = sparse_kv + metadata + selection + q_out
    dense_total = dense_kv + q_out
    return {
        "selected_tokens": sel_tokens, "dense_tokens": dense_tokens,
        "selected_pages": n_sel_pages, "total_pages": n_pages,
        "sparse_kv_bytes": sparse_kv, "metadata_bytes": metadata, "selection_bytes": selection,
        "sparse_total_bytes": sparse_total, "dense_total_bytes": dense_total,
        "bytes_ratio": sparse_total / max(dense_total, 1),
    }
