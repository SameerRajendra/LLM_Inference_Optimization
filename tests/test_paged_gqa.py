"""Correctness gate for the paged KV pool: fused FP8 append and ragged paged GQA decode (WP-1).

Runs on a GPU, or on CPU under the Triton interpreter:
    TRITON_INTERPRET=1 python -m pytest tests/test_paged_gqa.py -v
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triton_kernels.gqa_decode import KV_PAGE_TOKENS, quantize_kv_fp8  # noqa: E402
from triton_kernels.paged_gqa_decode import paged_gqa_decode  # noqa: E402
from triton_kernels.paged_kv import (  # noqa: E402
    allocate_paged_kv_cache, apply_rope, apply_rope_and_cache, reshape_and_cache,
    reshape_and_cache_fp8)

HKV = 2
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" and not INTERPRET, reason="needs a GPU or TRITON_INTERPRET=1")

# (B, seq_lens): every length from the WP-1 matrix alone, and two ragged batches of 4.
SEQ_CASES = [(1, [n]) for n in (1, 63, 64, 65, 257, 1025)] + [
    (4, [1, 63, 64, 65]), (4, [257, 1025, 1, 64])]


def _randn(g, *shape):
    return torch.randn(*shape, generator=g).half().to(DEVICE)


def _build_pool(seq_lens, D, fp8, block_size=64, seed=0, spare_blocks=3):
    """Fill a pool with random K/V for each sequence on shuffled physical blocks.

    Returns the pool, the block table and the logical fp16 K/V per sequence."""
    g = torch.Generator().manual_seed(seed)
    blocks_per_seq = [-(-n // block_size) for n in seq_lens]
    num_blocks = sum(blocks_per_seq) + spare_blocks
    perm = torch.randperm(num_blocks, generator=g).int()
    table = torch.zeros(len(seq_lens), max(blocks_per_seq), dtype=torch.int32)
    k_cache, v_cache, ks, vs = allocate_paged_kv_cache(num_blocks, block_size, HKV, D, fp8,
                                                       DEVICE)
    logical, used = [], 0
    for b, n in enumerate(seq_lens):
        table[b, :blocks_per_seq[b]] = perm[used:used + blocks_per_seq[b]]
        used += blocks_per_seq[b]
        k, v = _randn(g, n, HKV, D), _randn(g, n, HKV, D)
        pos = torch.arange(n)
        slots = (table[b, pos // block_size].long() * block_size + pos % block_size).to(DEVICE)
        if fp8:
            reshape_and_cache_fp8(k, v, k_cache, v_cache, ks, vs, slots)
        else:
            reshape_and_cache(k, v, k_cache, v_cache, slots)
        logical.append((k, v))
    return (k_cache, v_cache, ks, vs), table.to(DEVICE), logical


def _dequantize_pool(bits, scale):
    s = scale.view(bits.shape[0], -1, bits.shape[2]).repeat_interleave(KV_PAGE_TOKENS, dim=1)
    return bits.view(torch.float8_e4m3fn).float() * s[..., None]


def _gather(cache, table, b, n):
    """Logical [n, Hkv, D] view of sequence b, read back through the block table."""
    bs = cache.shape[1]
    return cache[table[b, :-(-n // bs)].long()].reshape(-1, *cache.shape[2:])[:n]


def _reference(q, k_pool, v_pool, table, seq_lens):
    """fp32 SDPA (native GQA) per sequence over K/V gathered from the shuffled pool."""
    outs = []
    for b, n in enumerate(seq_lens):
        k = _gather(k_pool, table, b, n).float().transpose(0, 1)
        v = _gather(v_pool, table, b, n).float().transpose(0, 1)
        o = F.scaled_dot_product_attention(q[b].float().unsqueeze(1), k, v, enable_gqa=True)
        outs.append(o.squeeze(1))
    return torch.stack(outs)


@pytest.mark.parametrize("num_splits", [1, 4])
@pytest.mark.parametrize("fp8", [False, True], ids=["fp16", "fp8"])
@pytest.mark.parametrize("G", [1, 4, 8])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("B,seq_lens", SEQ_CASES)
def test_paged_decode_matches_reference(B, seq_lens, D, G, fp8, num_splits):
    (kc, vc, ks, vs), table, _ = _build_pool(seq_lens, D, fp8)
    q = _randn(torch.Generator().manual_seed(1), B, G * HKV, D)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    out = paged_gqa_decode(q, kc, vc, table, lens, k_scale=ks, v_scale=vs,
                           max_seq_len=max(seq_lens), num_splits=num_splits)
    if fp8:  # isolate the kernel from quantisation error
        kc, vc = _dequantize_pool(kc, ks), _dequantize_pool(vc, vs)
    ref = _reference(q, kc, vc, table, seq_lens)
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=0)


@pytest.mark.parametrize("B,seq_lens", [(1, [1025]), (4, [1, 17, 64, 513])])
def test_paged_decode_block_size_16(B, seq_lens):
    (kc, vc, _, _), table, _ = _build_pool(seq_lens, 128, False, block_size=16)
    q = _randn(torch.Generator().manual_seed(1), B, 4 * HKV, 128)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    out = paged_gqa_decode(q, kc, vc, table, lens, num_splits=4)
    torch.testing.assert_close(out.float(), _reference(q, kc, vc, table, seq_lens),
                               atol=2e-3, rtol=0)


def test_paged_decode_default_splits_and_no_max_len():
    seq_lens = [17, 64, 513, 4097]
    (kc, vc, ks, vs), table, _ = _build_pool(seq_lens, 128, True)
    q = _randn(torch.Generator().manual_seed(1), 4, 4 * HKV, 128)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    out = paged_gqa_decode(q, kc, vc, table, lens, k_scale=ks, v_scale=vs)
    ref = _reference(q, _dequantize_pool(kc, ks), _dequantize_pool(vc, vs), table, seq_lens)
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=0)


def test_empty_sequence_gives_zero():
    seq_lens = [0, 65]
    (kc, vc, _, _), table, _ = _build_pool([1, 65], 64, False)
    q = _randn(torch.Generator().manual_seed(1), 2, 4 * HKV, 64)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    out = paged_gqa_decode(q, kc, vc, table, lens, num_splits=2)
    assert torch.isfinite(out).all() and out[0].abs().max() == 0
    torch.testing.assert_close(out[1:].float(), _reference(q[1:], kc, vc, table[1:], [65]),
                               atol=2e-3, rtol=0)


def test_fp16_append_is_exact():
    seq_lens = [1, 63, 64, 65, 257]
    (kc, vc, _, _), table, logical = _build_pool(seq_lens, 128, False, block_size=16)
    for b, (k, v) in enumerate(logical):
        assert torch.equal(_gather(kc, table, b, len(k)), k)
        assert torch.equal(_gather(vc, table, b, len(v)), v)


@pytest.mark.parametrize("D", [64, 128])
def test_fp8_prefill_matches_contiguous_quantizer(D):
    """Appending whole sequences reproduces quantize_kv_fp8 bit for bit: same pages, scales, bytes.

    The reference runs on CPU because CUDA torch divides by a Python scalar via a reciprocal
    multiply (1 ulp off IEEE); the kernel uses IEEE division (div_rn)."""
    seq_lens = [1, 63, 64, 65, 257]
    (kc, vc, ks, vs), table, logical = _build_pool(seq_lens, D, True, block_size=128)
    for b, (k, v) in enumerate(logical):
        n = len(k)
        kq, vq, kss, vss = quantize_kv_fp8(k[None].cpu(), v[None].cpu())
        # block_size 128 = two scale pages per block: logical page p -> (block, p % 2).
        pages = [(table[b, p // 2].long() * 2 + p % 2).item() for p in range(kss.shape[1])]
        assert torch.equal(ks.view(-1, HKV)[pages].cpu(), kss[0])
        assert torch.equal(vs.view(-1, HKV)[pages].cpu(), vss[0])
        assert torch.equal(_gather(kc, table, b, n).cpu(), kq[0])
        assert torch.equal(_gather(vc, table, b, n).cpu(), vq[0])


def test_fp8_incremental_append_rescales_pages():
    """Token-by-token decode appends with growing magnitude force every page to re-encode."""
    D, n = 128, 150
    g = torch.Generator().manual_seed(3)
    growth = torch.linspace(0.1, 4.0, n).view(n, 1, 1)
    k = (torch.randn(n, HKV, D, generator=g) * growth).half().to(DEVICE)
    v = (torch.randn(n, HKV, D, generator=g) * growth).half().to(DEVICE)
    kc, vc, ks, vs = allocate_paged_kv_cache(4, 64, HKV, D, True, DEVICE)
    table = torch.tensor([[2, 0, 3]], dtype=torch.int32, device=DEVICE)
    for t in range(n):
        slot = torch.tensor([table[0, t // 64].item() * 64 + t % 64], device=DEVICE)
        reshape_and_cache_fp8(k[t:t + 1], v[t:t + 1], kc, vc, ks, vs, slot)

    for x, cache, scale in ((k, kc, ks), (v, vc, vs)):
        absmax = torch.stack([x[p * 64:(p + 1) * 64].float().abs().amax(dim=(0, 2))
                              for p in range(3)]).cpu()
        # max commutes with the (monotone) division, so the running max is exactly the page's.
        assert torch.equal(scale.view(-1, HKV)[table[0].long()].cpu(), absmax / 448)
        # Each scale increase adds at most one e4m3 rounding; growth is monotone, so the
        # oldest token of a page sees <= 64 roundings, but each step's ratio is close to 1 and
        # the error grows much slower. Bound against the page scale (subnormal floor included).
        deq = _gather(_dequantize_pool(cache, scale), table, 0, n)
        page_scale = scale.view(-1, HKV)[table[0].long()].repeat_interleave(64, 0)[:n, :, None]
        err = (deq - x.float()).abs()
        assert (err <= 0.25 * x.float().abs() + 4 * page_scale).all(), err.max()

    # The decode kernel reads the re-encoded pool exactly as dequantised.
    q = _randn(g, 1, 4 * HKV, D)
    lens = torch.tensor([n], dtype=torch.int32, device=DEVICE)
    out = paged_gqa_decode(q, kc, vc, table, lens, k_scale=ks, v_scale=vs, num_splits=2)
    ref = _reference(q, _dequantize_pool(kc, ks), _dequantize_pool(vc, vs), table, [n])
    # V grows to ~|12| here, so the output's fp16 rounding (half ulp = 2^-11 relative) shows.
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=2.0 ** -10)
    # Worst case for re-encoding (the scale grows on every append) costs < 2x the RMS error of
    # quantising the same tokens in one shot, and ends on the same scales.
    one = allocate_paged_kv_cache(4, 64, HKV, D, True, DEVICE)
    slots = torch.tensor([table[0, t // 64].item() * 64 + t % 64 for t in range(n)],
                         device=DEVICE)
    reshape_and_cache_fp8(k, v, *one, slots)
    assert torch.equal(one[2], ks) and torch.equal(one[3], vs)
    rms = lambda c, s, x: (_gather(_dequantize_pool(c, s), table, 0, n)  # noqa: E731
                           - x.float()).pow(2).mean().sqrt().item()
    assert rms(kc, ks, k) < 2 * rms(one[0], one[2], k)
    assert rms(vc, vs, v) < 2 * rms(one[1], one[3], v)


def test_fp8_append_skips_negative_slots_and_zero_pages():
    kc, vc, ks, vs = allocate_paged_kv_cache(2, 64, HKV, 64, True, DEVICE)
    k = torch.zeros(3, HKV, 64, dtype=torch.float16, device=DEVICE)
    k[2] = 1.0
    slots = torch.tensor([0, 64, -1], device=DEVICE)  # token 2 is padding
    reshape_and_cache_fp8(k, k, kc, vc, ks, vs, slots)
    assert ks.abs().max() == 0 and vs.abs().max() == 0, "all-zero pages keep scale 0"
    assert kc.max() == 0
    small = torch.full((1, HKV, 64), 1e-3, dtype=torch.float16, device=DEVICE)
    reshape_and_cache_fp8(small, small, kc, vc, ks, vs, torch.tensor([1], device=DEVICE))
    deq = _dequantize_pool(kc, ks)[0, 1]
    torch.testing.assert_close(deq, small[0].float(), rtol=2 ** -4, atol=0)


def test_rope_and_cache_matches_unfused():
    D, n, max_pos = 64, 70, 128
    g = torch.Generator().manual_seed(5)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    ang = torch.arange(max_pos).float()[:, None] * inv_freq[None]
    cos, sin = ang.cos().to(DEVICE), ang.sin().to(DEVICE)
    k, v = _randn(g, n, HKV, D), _randn(g, n, HKV, D)
    positions = torch.arange(n, device=DEVICE) + 7
    slots = torch.arange(n, device=DEVICE)
    a = allocate_paged_kv_cache(2, 64, HKV, D, True, DEVICE)
    b = allocate_paged_kv_cache(2, 64, HKV, D, True, DEVICE)
    k_rot = apply_rope_and_cache(k, v, a[0], a[1], slots, cos, sin, positions, a[2], a[3])
    reshape_and_cache_fp8(apply_rope(k, cos, sin, positions), v, *b, slots)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    # Rotation preserves each (i, i + D/2) pair's norm.
    pair = lambda x: x[..., :D // 2].float() ** 2 + x[..., D // 2:].float() ** 2  # noqa: E731
    torch.testing.assert_close(pair(k_rot), pair(k), atol=1e-2, rtol=1e-2)
