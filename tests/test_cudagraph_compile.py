"""WP-2 gate: custom ops, torch.compile without graph breaks, CUDA-graph capture and replay.

CPU (op schemas, fake impls, fullgraph compile, block recycling):
    TRITON_INTERPRET=1 python -m pytest tests/test_cudagraph_compile.py -v
GPU (adds CUDA-graph capture/replay with changing seq_lens / block_table contents):
    python -m pytest tests/test_cudagraph_compile.py -v
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparse_kv import ops  # noqa: E402  (registers torch.ops.sparse_kv.*)
from triton_kernels.gqa_decode import KV_PAGE_TOKENS, quantize_kv_fp8  # noqa: E402
from triton_kernels.paged_kv import (  # noqa: E402
    BlockAllocator, allocate_paged_kv_cache, apply_rope)

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HKV, G, D, BS = 2, 4, 128, 64
HQ = HKV * G

pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" and not INTERPRET, reason="needs a GPU or TRITON_INTERPRET=1")
needs_graphs = pytest.mark.skipif(
    DEVICE != "cuda" or INTERPRET, reason="CUDA graphs need a GPU (and no Triton interpreter)")
FP8 = pytest.mark.parametrize("fp8", [False, True], ids=["fp16", "fp8"])


def _randn(g, *shape, scale=1.0):
    return (torch.randn(*shape, generator=g) * scale).half().to(DEVICE)


class _Pool:
    """A paged pool plus allocator; sequences are admitted with random K/V and evicted."""

    def __init__(self, num_blocks, fp8, max_blocks, B):
        self.fp8 = fp8
        self.kc, self.vc, self.ks, self.vs = allocate_paged_kv_cache(
            num_blocks, BS, HKV, D, fp8, DEVICE)
        self.alloc = BlockAllocator(num_blocks, scales=(self.ks, self.vs))
        self.table = torch.zeros(B, max_blocks, dtype=torch.int32)
        self.lens = [0] * B
        self.blocks = [[] for _ in range(B)]

    def admit(self, b, n, g, scale=1.0):
        assert not self.blocks[b]
        self.blocks[b] = self.alloc.allocate(-(-n // BS))
        self.table[b].zero_()
        self.table[b, :len(self.blocks[b])] = torch.tensor(self.blocks[b], dtype=torch.int32)
        self.lens[b] = n
        pos = torch.arange(n)
        slots = (self.table[b, pos // BS].long() * BS + pos % BS).to(DEVICE)
        k, v = _randn(g, n, HKV, D, scale=scale), _randn(g, n, HKV, D, scale=scale)
        ops.reshape_and_cache(k, v, self.kc, self.vc, slots, self.ks, self.vs)

    def evict(self, b):
        self.alloc.free(self.blocks[b])
        self.blocks[b] = []

    def logical(self, b):
        """[n, Hkv, D] fp32 K/V of sequence b as the kernel sees them (dequantised for FP8)."""
        n = self.lens[b]
        idx = self.table[b, :-(-n // BS)].long().to(DEVICE)
        out = []
        for c, s in ((self.kc, self.ks), (self.vc, self.vs)):
            x = c[idx]
            if self.fp8:
                sc = s[idx].repeat_interleave(KV_PAGE_TOKENS, dim=1)
                x = x.view(torch.float8_e4m3fn).float() * sc[..., None]
            out.append(x.reshape(-1, HKV, D)[:n].float())
        return out


def _sdpa_reference(q, pool):
    outs = []
    for b, n in enumerate(pool.lens):
        k, v = pool.logical(b)
        o = F.scaled_dot_product_attention(q[b].float().unsqueeze(1), k.transpose(0, 1),
                                           v.transpose(0, 1), enable_gqa=True)
        outs.append(o.squeeze(1))
    return torch.stack(outs)


def _small_pool(fp8, seq_lens=(70, 1, 130), seed=0):
    g = torch.Generator().manual_seed(seed)
    max_len = max(seq_lens)
    pool = _Pool(len(seq_lens) * -(-max_len // BS) + 2, fp8, -(-max_len // BS), len(seq_lens))
    for b, n in enumerate(seq_lens):
        pool.admit(b, n, g)
    q = _randn(g, len(seq_lens), HQ, D)
    return pool, q, g


# ----------------------------------------------------------------------- registration + schema

def test_ops_registered():
    for name in ("fused_gqa_v4", "fused_gqa_v4_fp8", "triton_gqa_decode", "paged_gqa_decode",
                 "reshape_and_cache", "reshape_and_cache_fp8"):
        assert hasattr(torch.ops.sparse_kv, name), name


@FP8
def test_opcheck_paged_decode_and_append(fp8):
    pool, q, g = _small_pool(fp8)
    bt, sl = pool.table.to(DEVICE), torch.tensor(pool.lens, dtype=torch.int32, device=DEVICE)
    checks = ("test_schema", "test_faketensor")
    torch.library.opcheck(torch.ops.sparse_kv.paged_gqa_decode.default,
                          (q, pool.kc, pool.vc, bt, sl, D ** -0.5, pool.ks, pool.vs, 256, 2),
                          test_utils=checks)
    k_new, v_new = _randn(g, 3, HKV, D), _randn(g, 3, HKV, D)
    slots = torch.tensor([70, -1, 129], dtype=torch.long, device=DEVICE)
    if fp8:
        torch.library.opcheck(torch.ops.sparse_kv.reshape_and_cache_fp8.default,
                              (k_new, v_new, pool.kc, pool.vc, pool.ks, pool.vs, slots),
                              test_utils=checks)
    else:
        torch.library.opcheck(torch.ops.sparse_kv.reshape_and_cache.default,
                              (k_new, v_new, pool.kc, pool.vc, slots), test_utils=checks)


@FP8
def test_paged_op_matches_reference(fp8):
    pool, q, _ = _small_pool(fp8)
    bt, sl = pool.table.to(DEVICE), torch.tensor(pool.lens, dtype=torch.int32, device=DEVICE)
    out = ops.paged_gqa_decode(q, pool.kc, pool.vc, bt, sl, max_seq_len=256,
                               k_scale=pool.ks, v_scale=pool.vs)
    torch.testing.assert_close(out.float(), _sdpa_reference(q, pool), atol=1e-2, rtol=1e-2)


def test_paged_op_requires_max_seq_len():
    with pytest.raises(TypeError):
        ops.paged_gqa_decode(None, None, None, None, None)  # max_seq_len is not optional


# ------------------------------------------------------------------- torch.compile(fullgraph)

def _decode_layer(q, k_new, v_new, kc, vc, ks, vs, block_table, positions, cos, sin, w_o,
                  max_seq_len):
    """One attention layer's decode: RoPE, append to the pool, paged attention, o_proj."""
    seq_lens = (positions + 1).int()
    blk = block_table.gather(1, (positions // BS).view(-1, 1)).view(-1).long()
    slots = blk * BS + positions % BS
    q = apply_rope(q, cos, sin, positions)
    k_new = apply_rope(k_new, cos, sin, positions)
    ops.reshape_and_cache(k_new, v_new, kc, vc, slots, ks, vs)
    o = ops.paged_gqa_decode(q, kc, vc, block_table, seq_lens, max_seq_len, k_scale=ks,
                             v_scale=vs)
    return F.linear(o.reshape(o.shape[0], -1), w_o)


def _layer_inputs(fp8, seed=1):
    pool, _, g = _small_pool(fp8, seed=seed)
    B = len(pool.lens)
    q, k_new, v_new = _randn(g, B, HQ, D), _randn(g, B, HKV, D), _randn(g, B, HKV, D)
    positions = torch.tensor(pool.lens, dtype=torch.long, device=DEVICE)  # next free slot
    ang = torch.arange(512, dtype=torch.float32)[:, None] * 0.01 * torch.arange(1, D // 2 + 1)
    cos, sin = ang.cos().to(DEVICE), ang.sin().to(DEVICE)
    w_o = _randn(g, 64, HQ * D, scale=0.02)
    return pool, (q, k_new, v_new, pool.kc, pool.vc, pool.ks, pool.vs, pool.table.to(DEVICE),
                  positions, cos, sin, w_o, 256)


def _cloned(args):
    return tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)


@FP8
def test_compile_fullgraph_no_graph_breaks(fp8):
    torch._dynamo.reset()
    _, args = _layer_inputs(fp8)
    explanation = torch._dynamo.explain(_decode_layer)(*_cloned(args))
    assert explanation.graph_break_count == 0, explanation.break_reasons

    eager_args, compiled_args = _cloned(args), _cloned(args)
    ref = _decode_layer(*eager_args)
    out = torch.compile(_decode_layer, backend="aot_eager", fullgraph=True)(*compiled_args)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    for i in (3, 4, 5, 6):  # the pool mutation made it through functionalization
        if eager_args[i] is not None:
            assert torch.equal(compiled_args[i], eager_args[i])


def test_compile_triton_gqa_decode_fullgraph():
    torch._dynamo.reset()
    g = torch.Generator().manual_seed(2)
    q, k, v = _randn(g, 2, HQ, D), _randn(g, 2, 192, HKV, D), _randn(g, 2, 192, HKV, D)
    kq, vq, ks, vs = quantize_kv_fp8(k, v)

    def fn(q, k, v, kq, vq, ks, vs):
        return ops.triton_gqa_decode(q, k, v) + ops.triton_gqa_decode(q, kq, vq, k_scale=ks,
                                                                      v_scale=vs)

    assert torch._dynamo.explain(fn)(q, k, v, kq, vq, ks, vs).graph_break_count == 0
    out = torch.compile(fn, backend="aot_eager", fullgraph=True)(q, k, v, kq, vq, ks, vs)
    torch.testing.assert_close(out, fn(q, k, v, kq, vq, ks, vs), atol=0, rtol=0)


@pytest.mark.skipif(DEVICE != "cuda" or INTERPRET, reason="CUDA extension needs a GPU")
def test_fused_v4_ops():
    try:
        from sparse_kv import _C  # noqa: F401
    except ImportError:
        pytest.skip("sparse_kv._C not built")
    g = torch.Generator().manual_seed(3)
    q = _randn(g, 2, 32, D)
    k, v = _randn(g, 2, 256, 8, D), _randn(g, 2, 256, 8, D)
    ref = ops.triton_gqa_decode(q, k, v)
    torch.testing.assert_close(ops.fused_gqa_v4(q, k, v), ref, atol=2e-3, rtol=2e-3)
    kq, vq, ks, vs = quantize_kv_fp8(k, v)
    torch.testing.assert_close(ops.fused_gqa_v4_fp8(q, kq, vq, ks, vs),
                               ops.triton_gqa_decode(q, kq, vq, k_scale=ks, v_scale=vs),
                               atol=2e-3, rtol=2e-3)
    fn = torch.compile(lambda q, k, v: ops.fused_gqa_v4(q, k, v), backend="aot_eager",
                       fullgraph=True)
    torch.testing.assert_close(fn(q, k, v), ops.fused_gqa_v4(q, k, v), atol=0, rtol=0)


# ---------------------------------------------------------------------------- block recycling

def test_recycled_block_scales_are_zeroed():
    g = torch.Generator().manual_seed(4)
    pool = _Pool(4, True, 2, 2)
    pool.admit(0, 128, g, scale=50.0)            # large values -> large page scales
    big = pool.ks[pool.blocks[0]].clone()
    assert (big > 0.05).all()
    recycled = list(pool.blocks[0])
    pool.evict(0)
    assert (pool.ks[recycled] == 0).all() and (pool.vs[recycled] == 0).all()

    pool.admit(1, 128, g, scale=0.01)            # LIFO free list: gets the same blocks
    assert sorted(pool.blocks[1]) == sorted(recycled)
    small = pool.ks[pool.blocks[1]]
    assert (small < big / 100).all(), "recycled block kept the previous owner's scale"
    k, _ = pool.logical(1)
    # e4m3 with a fresh scale: relative error ~2^-4; a stale 5000x scale would flush to 0.
    assert k.abs().max() > 0.02


# ----------------------------------------------------------------- CUDA-graph capture/replay

@needs_graphs
@FP8
def test_cudagraph_replay_tracks_seq_lens_and_block_table(fp8):
    """Capture once, then change seq_lens, block_table and q contents (with blocks recycled
    between sequences) and replay: every replay must equal an eager call on the same inputs."""
    B, max_seq_len = 4, 1024
    max_blocks = max_seq_len // BS
    g = torch.Generator().manual_seed(5)
    pool = _Pool(B * max_blocks + 4, fp8, max_blocks, B)

    q_s = torch.empty(B, HQ, D, dtype=torch.float16, device=DEVICE)
    bt_s = torch.zeros(B, max_blocks, dtype=torch.int32, device=DEVICE)
    sl_s = torch.zeros(B, dtype=torch.int32, device=DEVICE)

    def load(lens):
        for b, n in enumerate(lens):
            if pool.blocks[b] and pool.lens[b] != n:
                pool.evict(b)
            if not pool.blocks[b]:
                pool.admit(b, n, g)
        q_s.copy_(_randn(g, B, HQ, D))
        bt_s.copy_(pool.table)
        sl_s.copy_(torch.tensor(pool.lens, dtype=torch.int32))

    def step():
        return ops.paged_gqa_decode(q_s, pool.kc, pool.vc, bt_s, sl_s, max_seq_len,
                                    k_scale=pool.ks, v_scale=pool.vs)

    load([1000, 64, 1, 513])
    ops.warmup_paged_gqa_decode(q_s, pool.kc, pool.vc, bt_s, sl_s, max_seq_len,
                                k_scale=pool.ks, v_scale=pool.vs)
    tuned = ops.autotune_cache_sizes()
    graph, out_s = ops.capture_cuda_graph(step, warmup_iters=1)
    assert ops.autotune_cache_sizes() == tuned
    ptrs = [t.data_ptr() for t in (q_s, bt_s, sl_s, out_s)]

    for lens in ([1000, 64, 1, 513], [3, 1024, 1, 700], [65, 129, 900, 1], [1024] * 4):
        load(lens)
        graph.replay()
        torch.cuda.synchronize()
        assert [t.data_ptr() for t in (q_s, bt_s, sl_s, out_s)] == ptrs
        assert torch.equal(out_s, step()), lens
        torch.testing.assert_close(out_s.float(), _sdpa_reference(q_s, pool),
                                   atol=1e-2, rtol=1e-2)
    assert ops.autotune_cache_sizes() == tuned


@needs_graphs
@FP8
def test_cudagraph_captures_append_and_decode(fp8):
    """A whole decode layer (RoPE + FP8/FP16 append + paged attention + o_proj) in one graph,
    replayed for several steps with positions advanced in-graph, matches eager step by step."""
    _, args = _layer_inputs(fp8, seed=6)
    eager, static = list(_cloned(args)), list(_cloned(args))
    g = torch.Generator().manual_seed(7)
    B = args[0].shape[0]

    def step():
        out = _decode_layer(*static)
        static[8].add_(1)  # positions
        return out

    # Warm-up (JIT + autotune at this exact B / max_seq_len) appends to the pool and advances
    # positions, so snapshot the mutable state and restore it before capturing.
    mutable = [t for t in static[:9] if t is not None]
    saved = [t.clone() for t in mutable]
    ops._side_stream_run(step, 2)
    for t, s in zip(mutable, saved):
        t.copy_(s)
    graph, out_s = ops.capture_cuda_graph(step, warmup_iters=0)

    for _ in range(5):
        new = (_randn(g, B, HQ, D), _randn(g, B, HKV, D), _randn(g, B, HKV, D))
        for i, x in enumerate(new):
            static[i].copy_(x)
            eager[i].copy_(x)
        graph.replay()
        ref = _decode_layer(*eager)
        eager[8].add_(1)
        torch.cuda.synchronize()
        assert torch.equal(out_s, ref)
    for i in (3, 4, 5, 6):
        if eager[i] is not None:
            assert torch.equal(static[i], eager[i])
