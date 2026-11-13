"""WP-4: Quest page metadata, page scorer, top-k selection and page-sparse paged decode.

CPU:  TRITON_INTERPRET=1 pytest tests/test_quest_sparse.py -v
GPU:  pytest tests/test_quest_sparse.py -v
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_kernels.paged_gqa_decode import paged_gqa_decode  # noqa: E402
from triton_kernels.paged_kv import (  # noqa: E402
    BlockAllocator, allocate_page_minmax, allocate_paged_kv_cache, reshape_and_cache,
    reshape_and_cache_fp8,
)
from triton_kernels.quest_sparse_decode import (  # noqa: E402
    pages_for_budget, paged_sparse_decode, quest_hbm_bytes, quest_page_scores,
    quest_select_pages, quest_sparse_decode,
)

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"
pytestmark = pytest.mark.skipif(not INTERPRET and not torch.cuda.is_available(),
                                reason="needs a GPU or TRITON_INTERPRET=1")


def cdiv(a, b):
    return (a + b - 1) // b


def build_pool(seq_lens, Hkv, D, block_size, fp8=False, seed=0, decode_steps=2):
    """Shuffled-block-table paged pool with Quest metadata. All but the last `decode_steps`
    tokens of each sequence go in as one prefill append; the rest arrive one decode step at a
    time for the whole batch (slot -1 for sequences too short to have that token), so the
    metadata is exercised both in bulk and incrementally."""
    g = torch.Generator().manual_seed(seed)
    B = len(seq_lens)
    pages = [cdiv(s, block_size) for s in seq_lens]
    num_blocks = sum(pages) + 3  # a few blocks nobody owns
    perm = torch.randperm(num_blocks, generator=g).tolist()
    max_pages = max(max(pages), 1)
    bt = torch.full((B, max_pages), num_blocks - 1, dtype=torch.int32)
    owned, i = [], 0
    for b, n in enumerate(pages):
        bt[b, :n] = torch.tensor(perm[i:i + n], dtype=torch.int32)
        owned.append(perm[i:i + n])
        i += n

    kc, vc, ks, vs = allocate_paged_kv_cache(num_blocks, block_size, Hkv, D, fp8=fp8,
                                             device=DEVICE)
    k_min, k_max = allocate_page_minmax(num_blocks, Hkv, D, device=DEVICE)
    keys = [torch.randn(s, Hkv, D, generator=g).half() for s in seq_lens]
    vals = [torch.randn(s, Hkv, D, generator=g).half() for s in seq_lens]

    def slots(b, positions):
        return [int(bt[b, p // block_size]) * block_size + p % block_size for p in positions]

    def append(k, v, sm):
        sm = torch.tensor(sm, dtype=torch.int32, device=DEVICE)
        k, v = k.to(DEVICE), v.to(DEVICE)
        if fp8:
            reshape_and_cache_fp8(k, v, kc, vc, ks, vs, sm, k_min=k_min, k_max=k_max)
        else:
            reshape_and_cache(k, v, kc, vc, sm, k_min=k_min, k_max=k_max)

    for b, s in enumerate(seq_lens):
        n = max(s - decode_steps, 0)
        if n:
            append(keys[b][:n], vals[b][:n], slots(b, range(n)))
    for step in range(decode_steps):
        k_t, v_t, sm = [], [], []
        for b, s in enumerate(seq_lens):
            pos = s - decode_steps + step
            if pos >= 0:
                k_t.append(keys[b][pos])
                v_t.append(vals[b][pos])
                sm += slots(b, [pos])
            else:  # too short to have this token: slot -1 must be skipped by the append
                k_t.append(torch.zeros(Hkv, D).half())
                v_t.append(torch.zeros(Hkv, D).half())
                sm.append(-1)
        append(torch.stack(k_t), torch.stack(v_t), sm)

    return dict(kc=kc, vc=vc, ks=ks, vs=vs, k_min=k_min, k_max=k_max, keys=keys, vals=vals,
                bt=bt.to(DEVICE), owned=owned, num_blocks=num_blocks,
                sl=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE))


def make_q(B, Hq, D, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, Hq, D, generator=g).half().to(DEVICE)


# ------------------------------------------------------------------------------- metadata

@pytest.mark.parametrize("fp8,block_size", [(False, 16), (False, 64), (True, 64)])
def test_page_minmax_matches_reference(fp8, block_size):
    seq_lens = [1, 37, 2 * block_size + 5]
    Hkv, D = 2, 64
    pool = build_pool(seq_lens, Hkv, D, block_size, fp8=fp8)
    k_min, k_max = pool["k_min"].cpu(), pool["k_max"].cpu()
    seen = set()
    for b, s in enumerate(seq_lens):
        for p in range(cdiv(s, block_size)):
            blk = int(pool["bt"][b, p])
            seen.add(blk)
            tok = pool["keys"][b][p * block_size:min((p + 1) * block_size, s)]
            assert torch.equal(k_min[blk], tok.amin(0)), (b, p)
            assert torch.equal(k_max[blk], tok.amax(0)), (b, p)
    for blk in set(range(pool["num_blocks"])) - seen:  # untouched blocks stay empty
        assert torch.isinf(k_min[blk]).all() and (k_min[blk] > 0).all()
        assert torch.isinf(k_max[blk]).all() and (k_max[blk] < 0).all()


def test_minmax_is_opt_in_and_allocator_resets():
    kc, vc, _, _ = allocate_paged_kv_cache(4, 16, 2, 64, device=DEVICE)
    k = torch.randn(5, 2, 64).half().to(DEVICE)
    sm = torch.arange(5, dtype=torch.int32, device=DEVICE)
    reshape_and_cache(k, k, kc, vc, sm)  # WP-1 call signature, no metadata
    assert torch.equal(kc[0, :5].cpu(), k.cpu())

    k_min, k_max = allocate_page_minmax(4, 2, 64, device=DEVICE)
    reshape_and_cache(k, k, kc, vc, sm, k_min=k_min, k_max=k_max)
    assert torch.isfinite(k_min[0]).all()
    alloc = BlockAllocator(4, page_bounds=[(k_min, k_max)])
    alloc.free(alloc.allocate(1))  # hands out block 0, then recycles it
    assert torch.isinf(k_min[0]).all() and torch.isinf(k_max[0]).all()


# --------------------------------------------------------------------------- page scorer

@pytest.mark.parametrize("G,recent", [(4, 1), (1, 2), (8, 0)])
def test_page_scores_match_quest_bound(G, recent):
    seq_lens, Hkv, D, bs = [0, 20, 150], 2, 64, 16
    pool = build_pool(seq_lens, Hkv, D, bs)
    q = make_q(len(seq_lens), Hkv * G, D)
    max_seq = max(seq_lens)
    scores = quest_page_scores(q, pool["k_min"], pool["k_max"], pool["bt"], pool["sl"], bs,
                               recent_pages=recent, max_seq_len=max_seq).cpu()
    assert scores.shape == (len(seq_lens), Hkv, cdiv(max_seq, bs))
    k_min, k_max, qf = pool["k_min"].cpu().float(), pool["k_max"].cpu().float(), q.cpu().float()
    for b, s in enumerate(seq_lens):
        n = cdiv(s, bs)
        for h in range(Hkv):
            qg = qf[b, h * G:(h + 1) * G]  # [G, D]
            for p in range(scores.shape[2]):
                got = scores[b, h, p].item()
                if p >= n:
                    assert got == -math.inf
                elif p == 0 or p >= n - recent:
                    assert got == math.inf
                else:
                    blk = int(pool["bt"][b, p])
                    ref = torch.maximum(qg * k_min[blk, h], qg * k_max[blk, h]).sum(-1).max()
                    assert abs(got - ref.item()) <= 1e-3 * max(1.0, abs(ref.item())), (b, h, p)
                    # The bound really bounds every key in the page.
                    tok = pool["keys"][b][p * bs:min((p + 1) * bs, s), h].float()
                    assert (qg @ tok.T).max().item() <= got + 1e-2


# ----------------------------------------------------------------------------- selection

@pytest.mark.parametrize("recent", [1, 3])
def test_sink_and_recent_pages_always_kept(recent):
    """Make the sink and the recent window the LEAST attractive pages; they must still be kept,
    and the remaining budget must go to the highest-scoring page (a planted needle)."""
    bs, Hkv, G, D = 16, 2, 2, 64
    seq_lens = [16 * 12 + 3, 16 * 4, 16 * 9]
    pool = build_pool(seq_lens, Hkv, D, bs, seed=3)
    B = len(seq_lens)
    q = torch.ones(B, Hkv * G, D).half().to(DEVICE)
    k_min, k_max = pool["k_min"], pool["k_max"]
    needle = {}
    for b, s in enumerate(seq_lens):
        n = cdiv(s, bs)
        for p in range(n):
            blk = int(pool["bt"][b, p])
            forced = p == 0 or p >= n - recent
            k_min[blk] = -4.0 if forced else -1.0
            k_max[blk] = -3.0 if forced else 0.5
        needle[b] = n // 2 if n > 1 + recent else None
        if needle[b] is not None:
            k_max[int(pool["bt"][b, needle[b]])] = 2.0
    top_k = 1 + recent + 1
    _, sel = quest_sparse_decode(q, pool["kc"], pool["vc"], pool["bt"], pool["sl"], k_min, k_max,
                                 top_k, recent_pages=recent, max_seq_len=max(seq_lens),
                                 return_selected=True)
    sel = sel.cpu()
    assert sel.shape == (B, Hkv, top_k)
    for b, s in enumerate(seq_lens):
        n = cdiv(s, bs)
        for h in range(Hkv):
            chosen = [int(x) for x in sel[b, h] if x >= 0]
            assert chosen == sorted(chosen) and len(set(chosen)) == len(chosen)
            assert all(0 <= x < n for x in chosen)
            assert len(chosen) == min(top_k, n)
            assert 0 in chosen, "sink page dropped"
            for p in range(max(n - recent, 0), n):
                assert p in chosen, "recent page {} dropped".format(p)
            if needle[b] is not None:
                assert needle[b] in chosen, "highest-scoring page not selected"


def test_short_sequences_fill_unused_slots_with_minus_one():
    scores = torch.tensor([[[math.inf, 1.0, -math.inf, -math.inf]]])
    sel = quest_select_pages(scores, 3)
    assert sel.tolist() == [[[0, 1, -1]]]
    assert pages_for_budget(0.125, 64 * 1024, 16, recent_pages=1) == 512
    assert pages_for_budget(0.01, 64, 16, recent_pages=2) == 3


# ------------------------------------------------------------------ budget 100% == dense

@pytest.mark.parametrize("fp8,block_size,G,D", [
    (False, 16, 4, 128), (False, 64, 8, 64), (False, 32, 1, 64),
    (True, 64, 4, 128), (True, 128, 2, 64),
])
def test_full_budget_matches_dense_paged(fp8, block_size, G, D):
    seq_lens = [0, 1, 77, 3 * block_size + 9, 2 * block_size]
    Hkv = 2
    pool = build_pool(seq_lens, Hkv, D, block_size, fp8=fp8, seed=7)
    q = make_q(len(seq_lens), Hkv * G, D, seed=8)
    max_seq = max(seq_lens)
    dense = paged_gqa_decode(q, pool["kc"], pool["vc"], pool["bt"], pool["sl"],
                             k_scale=pool["ks"], v_scale=pool["vs"], max_seq_len=max_seq)
    top_k = cdiv(max_seq, block_size)  # budget = 100%
    out, sel = quest_sparse_decode(q, pool["kc"], pool["vc"], pool["bt"], pool["sl"],
                                   pool["k_min"], pool["k_max"], top_k, recent_pages=1,
                                   k_scale=pool["ks"], v_scale=pool["vs"], max_seq_len=max_seq,
                                   return_selected=True)
    for b, s in enumerate(seq_lens):  # every valid page selected exactly once
        n = cdiv(s, block_size)
        for h in range(Hkv):
            assert sorted(int(x) for x in sel[b, h] if x >= 0) == list(range(n))
    tol = 1e-2 if fp8 else 1e-3
    err = (out.float() - dense.float()).abs().max().item()
    assert torch.isfinite(out).all()
    assert err < tol, "max |quest - dense| = {:.2e}".format(err)


# ---------------------------------------------------- stage 2 reads only the selected pages

def _dequant(pool, fp8):
    """Logical fp32 view of the pool: [num_blocks, block_size, Hkv, D] for K and V."""
    if not fp8:
        return pool["kc"].cpu().float(), pool["vc"].cpu().float()
    nb, bs, Hkv, D = pool["kc"].shape
    per = bs // 64

    def deq(c, s):
        x = c.cpu().view(torch.float8_e4m3fn).float()
        s = s.cpu().view(nb, per, Hkv).repeat_interleave(64, dim=1)  # [nb, bs, Hkv]
        return x * s[..., None]
    return deq(pool["kc"], pool["ks"]), deq(pool["vc"], pool["vs"])


@pytest.mark.parametrize("fp8,block_size", [(False, 16), (True, 64)])
def test_sparse_decode_never_touches_unselected_pages(fp8, block_size):
    """Poison every unselected page and every slot past seq_len with NaN; the output must stay
    finite and equal exact attention over the selected tokens."""
    seq_lens, Hkv, G, D = [5 * block_size + 3, 2 * block_size, 7 * block_size], 2, 4, 64
    pool = build_pool(seq_lens, Hkv, D, block_size, fp8=fp8, seed=11)
    B = len(seq_lens)
    q = make_q(B, Hkv * G, D, seed=12)
    max_seq = max(seq_lens)
    scores = quest_page_scores(q, pool["k_min"], pool["k_max"], pool["bt"], pool["sl"],
                               block_size, recent_pages=1, max_seq_len=max_seq)
    sel = quest_select_pages(scores, 3)

    Kf, Vf = _dequant(pool, fp8)  # reference values, taken before poisoning
    nan = 0x7F if fp8 else float("nan")
    for b, s in enumerate(seq_lens):
        n = cdiv(s, block_size)
        for h in range(Hkv):
            keep = set(int(x) for x in sel[b, h] if x >= 0)
            for p in range(n):
                blk = int(pool["bt"][b, p])
                if p not in keep:
                    pool["kc"][blk, :, h] = nan
                    pool["vc"][blk, :, h] = nan
            tail = s % block_size
            if tail:
                blk = int(pool["bt"][b, n - 1])
                pool["kc"][blk, tail:, h] = nan
                pool["vc"][blk, tail:, h] = nan

    out = paged_sparse_decode(q, pool["kc"], pool["vc"], pool["bt"], pool["sl"], sel,
                              k_scale=pool["ks"], v_scale=pool["vs"]).cpu().float()
    assert torch.isfinite(out).all(), "stage 2 read an unselected (poisoned) page"

    qf = q.cpu().float()
    for b, s in enumerate(seq_lens):
        for h in range(Hkv):
            idx = []
            for p in (int(x) for x in sel[b, h] if x >= 0):
                blk = int(pool["bt"][b, p])
                idx += [(blk, t) for t in range(min(block_size, s - p * block_size))]
            k = torch.stack([Kf[blk, t, h] for blk, t in idx])
            v = torch.stack([Vf[blk, t, h] for blk, t in idx])
            att = torch.softmax(qf[b, h * G:(h + 1) * G] @ k.T * D ** -0.5, dim=-1)
            ref = att @ v
            err = (out[b, h * G:(h + 1) * G] - ref).abs().max().item()
            assert err < (2e-2 if fp8 else 2e-3), (b, h, err)


def test_hbm_accounting_scales_with_budget():
    seq_lens, Hkv, G, D, bs = [40 * 16, 13 * 16 + 5], 2, 2, 64, 16
    pool = build_pool(seq_lens, Hkv, D, bs, decode_steps=1)
    q = make_q(2, Hkv * G, D)
    max_seq = max(seq_lens)
    full = None
    for top_k in (4, 8, cdiv(max_seq, bs)):
        _, sel = quest_sparse_decode(q, pool["kc"], pool["vc"], pool["bt"], pool["sl"],
                                     pool["k_min"], pool["k_max"], top_k, max_seq_len=max_seq,
                                     return_selected=True)
        acc = quest_hbm_bytes(sel, pool["sl"], bs, Hkv * G, D, max_pages=cdiv(max_seq, bs))
        assert acc["selected_pages"] <= 2 * Hkv * top_k
        if top_k == 4:
            # every row has >= 4 pages: exactly 4 full pages per (b, h) except a partial tail
            assert acc["selected_pages"] == 2 * Hkv * 4
        full = acc
    assert full["selected_tokens"] == full["dense_tokens"]


# --------------------------------------------------------------- custom op + CUDA graph

def test_custom_ops_match_python_entry_points():
    from sparse_kv import ops
    seq_lens, Hkv, G, D, bs = [50, 130], 2, 4, 64, 16
    pool = build_pool(seq_lens, Hkv, D, bs, decode_steps=0)
    # Append one more token per sequence through the custom op path.
    k_new = torch.randn(2, Hkv, D).half().to(DEVICE)
    sm = torch.tensor([int(pool["bt"][b, s // bs]) * bs + s % bs for b, s in enumerate(seq_lens)],
                      dtype=torch.int32, device=DEVICE)
    k_min_ref, k_max_ref = pool["k_min"].clone(), pool["k_max"].clone()
    ops.reshape_and_cache(k_new, k_new, pool["kc"], pool["vc"], sm, k_min=pool["k_min"],
                          k_max=pool["k_max"])
    reshape_and_cache(k_new, k_new, pool["kc"].clone(), pool["vc"].clone(), sm,
                      k_min=k_min_ref, k_max=k_max_ref)
    assert torch.equal(pool["k_min"], k_min_ref) and torch.equal(pool["k_max"], k_max_ref)
    sl = pool["sl"] + 1
    q = make_q(2, Hkv * G, D)
    a = ops.quest_sparse_decode(q, pool["kc"], pool["vc"], pool["bt"], sl, pool["k_min"],
                                pool["k_max"], 4, max_seq_len=int(sl.max()), recent_pages=2)
    b = quest_sparse_decode(q, pool["kc"], pool["vc"], pool["bt"], sl, pool["k_min"],
                            pool["k_max"], 4, recent_pages=2, max_seq_len=int(sl.max()))
    assert torch.equal(a, b)


@pytest.mark.skipif(INTERPRET or not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("fp8", [False, True])
def test_cuda_graph_capture_and_replay(fp8):
    from sparse_kv import ops
    bs = 64
    seq_lens, Hkv, G, D = [700, 64 * 9, 3], 2, 4, 128
    pool = build_pool(seq_lens, Hkv, D, bs, fp8=fp8, seed=21)
    max_seq = 1024  # static bound, larger than any sequence
    q = make_q(3, Hkv * G, D)
    top_k = pages_for_budget(0.25, max_seq, bs)

    def step():
        return ops.quest_sparse_decode(q, pool["kc"], pool["vc"], pool["bt"], pool["sl"],
                                       pool["k_min"], pool["k_max"], top_k, max_seq_len=max_seq,
                                       k_scale=pool["ks"], v_scale=pool["vs"])

    graph, out = ops.capture_cuda_graph(step)
    q.copy_(make_q(3, Hkv * G, D, seed=99))  # new query, same static buffer
    graph.replay()
    torch.cuda.synchronize()
    eager = step()
    assert torch.equal(out, eager)
