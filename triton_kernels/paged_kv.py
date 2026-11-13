"""Paged KV cache: allocation and the fused quantize + scatter append (WP-1).

Pool contract (shared with paged_gqa_decode.py):
    k_cache, v_cache  [num_blocks, block_size, Hkv, D]  fp16, or uint8 holding e4m3 bits
    k_scale, v_scale  [num_blocks, block_size // KV_PAGE_TOKENS, Hkv] fp32 (FP8 only;
                      [num_blocks, Hkv] is accepted when block_size == KV_PAGE_TOKENS)
    slot_mapping      [num_tokens] int, slot = physical_block * block_size + offset; < 0 skips

FP8 scales: one per (64-token page, kv head), equal to the largest |x| / 448 ever appended to
that page. A scale of 0 marks a page that is empty (or all zero); allocate scales with zeros,
and zero them again whenever a block is recycled (reset_block_scales / BlockAllocator.free).
Unlike quantize_kv_fp8, an all-zero page keeps scale 0, so a later small token still gets the
full e4m3 range instead of being stuck with scale 1.

Appending to a partly filled page can raise its scale. The bytes already in the page were
encoded with the old scale, so they are re-encoded with the new one first (old / new <= 1, so
this never saturates; it costs at most one extra e4m3 rounding per scale increase). Worst case,
a page whose scale grows on every append, measures ~1.7x the RMS error of quantising the same
tokens in one shot (tests/test_paged_gqa.py); prefill of whole pages re-encodes nothing.
The append is three launches, ordered by the stream:
    1. absmax   per token, atomic max of |x| / 448 into the page scale
    2. rescale  one program per touched (page, head) re-encodes the page if its scale grew
    3. store    every token quantised with its page's final scale and scattered to its slot
Nothing syncs with the host, so the append can be captured in a CUDA graph.

Quest page metadata (WP-4, opt-in): pass k_min/k_max [num_blocks, Hkv, D] fp16 (from
allocate_page_minmax) to reshape_and_cache* / apply_rope_and_cache and every append also folds
its fp16 keys (post-RoPE, pre-quantisation) into a channelwise running min/max per physical
block. Empty blocks hold k_min = +inf, k_max = -inf; reset them with reset_block_minmax (or
BlockAllocator(page_bounds=...)) when a block is recycled. The update is a scatter-reduce of
min/max, which is order independent, so it is deterministic, needs no host sync and can be
captured in a CUDA graph. Callers that do not pass the tensors get exactly the WP-1 behaviour.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .gqa_decode import E4M3_MAX, KV_PAGE_TOKENS

_E4M3 = tl.constexpr(E4M3_MAX)  # globals read inside @jit code must be constexpr


def allocate_paged_kv_cache(num_blocks, block_size, num_kv_heads, head_dim, fp8=False,
                            device="cuda"):
    """Returns (k_cache, v_cache, k_scale, v_scale); the scales are None for an fp16 pool."""
    shape = (num_blocks, block_size, num_kv_heads, head_dim)
    if not fp8:
        k = torch.zeros(shape, dtype=torch.float16, device=device)
        return k, torch.zeros_like(k), None, None
    assert block_size % KV_PAGE_TOKENS == 0, "FP8 pool needs block_size % 64 == 0"
    k = torch.zeros(shape, dtype=torch.uint8, device=device)
    scale_shape = (num_blocks, block_size // KV_PAGE_TOKENS, num_kv_heads)
    ks = torch.zeros(scale_shape, dtype=torch.float32, device=device)
    return k, torch.zeros_like(k), ks, torch.zeros_like(ks)


def reset_block_scales(blocks, *scales):
    """Zero the FP8 page scales of physical `blocks` (dim 0 of every scale tensor), marking their
    pages empty. Must run whenever a block is recycled: a stale scale would make the next owner
    quantise with the previous sequence's range (precision loss for smaller values), and the
    append would re-encode stale bytes as if they belonged to the page. No host sync."""
    scales = [s for s in scales if s is not None]
    if not scales:
        return
    idx = torch.as_tensor(blocks, dtype=torch.long).to(scales[0].device, non_blocking=True)
    if idx.numel() == 0:
        return
    for s in scales:
        s.index_fill_(0, idx, 0.0)


def allocate_page_minmax(num_blocks, num_kv_heads, head_dim, device="cuda"):
    """Quest page metadata: (k_min, k_max) [num_blocks, Hkv, D] fp16, initialised empty
    (k_min = +inf, k_max = -inf) so the first append to a block sets both exactly."""
    shape = (num_blocks, num_kv_heads, head_dim)
    k_min = torch.full(shape, float("inf"), dtype=torch.float16, device=device)
    k_max = torch.full(shape, float("-inf"), dtype=torch.float16, device=device)
    return k_min, k_max


def reset_block_minmax(blocks, k_min, k_max):
    """Mark physical `blocks` empty in the Quest metadata. Must run whenever a block is
    recycled, or the stale bounds of the previous owner would widen the new owner's bounds."""
    idx = torch.as_tensor(blocks, dtype=torch.long).to(k_min.device, non_blocking=True)
    if idx.numel() == 0:
        return
    k_min.index_fill_(0, idx, float("inf"))
    k_max.index_fill_(0, idx, float("-inf"))


def update_page_minmax(k_new, slot_mapping, k_min, k_max, block_size):
    """Fold [T, Hkv, D] fp16 keys into the running channelwise min/max of their physical blocks.
    Tokens with slot < 0 are skipped (their contribution is +inf / -inf, the reduce identity)."""
    T, Hkv, D = k_new.shape
    assert k_min.shape == k_max.shape and k_min.shape[1:] == (Hkv, D)
    assert k_min.dtype == torch.float16 and k_max.dtype == torch.float16
    if T == 0:
        return
    slots = slot_mapping.long()
    valid = (slots >= 0).view(T, 1, 1)
    idx = (slots.clamp_min(0) // block_size).view(T, 1, 1).expand(T, Hkv, D)
    k = k_new.to(torch.float16)
    k_min.scatter_reduce_(0, idx, torch.where(valid, k, float("inf")), reduce="amin")
    k_max.scatter_reduce_(0, idx, torch.where(valid, k, float("-inf")), reduce="amax")


class BlockAllocator:
    """Host-side free list over a pool's physical blocks.

    `scales` are the FP8 scale tensors whose dim 0 is the physical block (k_scale and v_scale,
    or one pair per layer); freed blocks get their scales zeroed so a recycled block starts
    empty. The data bytes are left alone: with scale 0 nothing re-encodes them, and decode masks
    every slot at or past seq_len. `page_bounds` are optional Quest (k_min, k_max) pairs (one per
    layer); freed blocks get them reset to empty."""

    def __init__(self, num_blocks, scales=(), page_bounds=()):
        self.num_blocks = num_blocks
        self.scales = [s for s in scales if s is not None]
        self.page_bounds = [pb for pb in page_bounds if pb is not None]
        self._free = list(range(num_blocks - 1, -1, -1))  # pop() hands out 0, 1, 2, ...

    @property
    def num_free(self):
        return len(self._free)

    def allocate(self, n):
        if n > len(self._free):
            raise RuntimeError("paged KV pool exhausted: need {}, {} free".format(
                n, len(self._free)))
        return [self._free.pop() for _ in range(n)]

    def free(self, blocks):
        blocks = [int(b) for b in blocks]
        reset_block_scales(blocks, *self.scales)
        for k_min, k_max in self.page_bounds:
            reset_block_minmax(blocks, k_min, k_max)
        self._free.extend(reversed(blocks))


@triton.jit
def _to_e4m3_bits(x):
    """fp32 -> e4m3 bits (uint8), round-to-nearest-even, saturating at +-448.

    Rounds onto the e4m3 grid in fp32 first so the final cast is exact. That matches the GPU's
    cvt.rn.satfinite, and sidesteps the Triton 3.3 interpreter's fp8 cast, which drops the carry
    when rounding up crosses an exponent (31.97 -> 16 instead of 32)."""
    x = tl.clamp(x, -_E4M3, _E4M3)
    # Normals: keep 3 of fp32's 23 mantissa bits, RNE by integer add; a carry bumps the exponent.
    bits = x.to(tl.int32, bitcast=True)
    bits = (bits + 0x7FFFF + ((bits >> 20) & 1)) & -0x100000
    normal = bits.to(tl.float32, bitcast=True)
    # Subnormals (|x| < 2^-6) sit on a 2^-9 grid; fp32 ulp at 2^14 is 2^-9, so the add rounds.
    sub = (tl.abs(x) + 16384.0) - 16384.0
    sub = tl.where(x < 0, -sub, sub)
    x = tl.where(tl.abs(x) < 0.015625, sub, normal)
    return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)


@triton.jit
def _kv_absmax_kernel(K, V, SM, KS_BITS, VS_BITS, Hkv,
                      D: tl.constexpr, PAGE: tl.constexpr, BLOCK_HKV: tl.constexpr):
    """Raise each touched page scale to cover this token. Scales are >= 0, so their fp32 bit
    patterns order like int32 and an integer atomic max is an exact float max."""
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(SM + t).to(tl.int64)
    if slot >= 0:
        heads = tl.arange(0, BLOCK_HKV)
        cols = tl.arange(0, D)
        ok = heads < Hkv
        row = ((t * Hkv + heads) * D)[:, None] + cols[None, :]
        k = tl.load(K + row, mask=ok[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + row, mask=ok[:, None], other=0.0).to(tl.float32)
        # div_rn: IEEE division (Triton's fp32 '/' is approximate on GPU), so the scales are
        # bit-identical to quantize_kv_fp8 run on CPU.
        e4m3 = tl.full([BLOCK_HKV], _E4M3, tl.float32)
        ks = tl.math.div_rn(tl.max(tl.abs(k), 1), e4m3)
        vs = tl.math.div_rn(tl.max(tl.abs(v), 1), e4m3)
        sc = (slot // PAGE) * Hkv + heads
        tl.atomic_max(KS_BITS + sc, ks.to(tl.int32, bitcast=True), mask=ok)
        tl.atomic_max(VS_BITS + sc, vs.to(tl.int32, bitcast=True), mask=ok)


@triton.jit
def _rescale_page(C, base, old, new, D: tl.constexpr, PAGE: tl.constexpr, Hkv):
    """Re-encode one (page, head) from scale `old` to the larger scale `new`."""
    rows = tl.arange(0, PAGE)
    cols = tl.arange(0, D)
    ptr = C + ((base + rows) * Hkv * D)[:, None] + cols[None, :]
    x = tl.load(ptr).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    tl.store(ptr, _to_e4m3_bits(x * (old / new)))


@triton.jit
def _kv_rescale_kernel(KC, VC, KS, VS, KEYS, OLD_KS, OLD_VS, Hkv,
                       D: tl.constexpr, PAGE: tl.constexpr):
    """Program (i, h) over the sorted page keys; only the first entry of each run of equal keys
    (the leader) touches the page, so every page is re-encoded at most once."""
    i = tl.program_id(0)
    h = tl.program_id(1)
    key = tl.load(KEYS + i)
    prev = tl.load(KEYS + tl.maximum(i - 1, 0))
    if (key >= 0) & ((i == 0) | (prev != key)):
        sc = key * Hkv + h
        base = key * PAGE  # first slot of the page
        old_k = tl.load(OLD_KS + i * Hkv + h)
        new_k = tl.load(KS + sc)
        if (old_k > 0) & (new_k > old_k):
            _rescale_page(KC + h * D, base, old_k, new_k, D, PAGE, Hkv)
        old_v = tl.load(OLD_VS + i * Hkv + h)
        new_v = tl.load(VS + sc)
        if (old_v > 0) & (new_v > old_v):
            _rescale_page(VC + h * D, base, old_v, new_v, D, PAGE, Hkv)


@triton.jit
def _kv_store_kernel(K, V, KC, VC, KS, VS, SM, Hkv,
                     D: tl.constexpr, PAGE: tl.constexpr, BLOCK_HKV: tl.constexpr,
                     FP8: tl.constexpr):
    """Scatter one token's [Hkv, D] K/V into its slot, quantised with the final page scale."""
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(SM + t).to(tl.int64)
    if slot >= 0:
        heads = tl.arange(0, BLOCK_HKV)
        cols = tl.arange(0, D)
        ok = heads < Hkv
        src = ((t * Hkv + heads) * D)[:, None] + cols[None, :]
        dst = ((slot * Hkv + heads) * D)[:, None] + cols[None, :]
        k = tl.load(K + src, mask=ok[:, None], other=0.0)
        v = tl.load(V + src, mask=ok[:, None], other=0.0)
        if FP8:
            sc = (slot // PAGE) * Hkv + heads
            ks = tl.load(KS + sc, mask=ok, other=1.0)
            vs = tl.load(VS + sc, mask=ok, other=1.0)
            # Scale 0 means the page is all zero, so every value encodes as 0.
            one = tl.full([BLOCK_HKV], 1.0, tl.float32)
            inv_k = tl.where(ks > 0, tl.math.div_rn(one, ks), 0.0)
            inv_v = tl.where(vs > 0, tl.math.div_rn(one, vs), 0.0)
            kq = _to_e4m3_bits(k.to(tl.float32) * inv_k[:, None])
            vq = _to_e4m3_bits(v.to(tl.float32) * inv_v[:, None])
            tl.store(KC + dst, kq, mask=ok[:, None])
            tl.store(VC + dst, vq, mask=ok[:, None])
        else:
            tl.store(KC + dst, k, mask=ok[:, None])
            tl.store(VC + dst, v, mask=ok[:, None])


def _check(k_new, v_new, k_cache, v_cache, slot_mapping):
    assert k_new.dim() == 3 and v_new.shape == k_new.shape, "k_new/v_new must be [T, Hkv, D]"
    assert k_cache.dim() == 4 and v_cache.shape == k_cache.shape
    assert k_new.shape[1:] == k_cache.shape[2:], "token [Hkv, D] must match the pool"
    assert k_new.dtype == torch.float16 and slot_mapping.shape == (k_new.shape[0],)
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    return k_new.contiguous(), v_new.contiguous(), slot_mapping.contiguous()


def _maybe_minmax(k_new, slot_mapping, k_cache, k_min, k_max):
    if k_min is None and k_max is None:
        return
    assert k_min is not None and k_max is not None, "pass both k_min and k_max"
    assert k_min.shape == (k_cache.shape[0],) + tuple(k_cache.shape[2:]), \
        "k_min/k_max must be [num_blocks, Hkv, D]"
    update_page_minmax(k_new, slot_mapping, k_min, k_max, k_cache.shape[1])


def reshape_and_cache(k_new, v_new, k_cache, v_cache, slot_mapping, k_min=None, k_max=None):
    """fp16 pool: scatter [T, Hkv, D] K/V into their slots. k_min/k_max (optional) are the
    Quest page bounds, updated in place."""
    k_new, v_new, slot_mapping = _check(k_new, v_new, k_cache, v_cache, slot_mapping)
    assert k_cache.dtype == torch.float16
    T, Hkv, D = k_new.shape
    if T == 0:
        return
    _maybe_minmax(k_new, slot_mapping, k_cache, k_min, k_max)
    _kv_store_kernel[(T,)](k_new, v_new, k_cache, v_cache, None, None, slot_mapping, Hkv,
                           D=D, PAGE=KV_PAGE_TOKENS,
                           BLOCK_HKV=triton.next_power_of_2(Hkv), FP8=False)


def reshape_and_cache_fp8(k_new, v_new, k_cache, v_cache, k_scale, v_scale, slot_mapping,
                          k_min=None, k_max=None):
    """FP8 pool: quantise [T, Hkv, D] fp16 K/V to e4m3, scatter to their slots, and keep the
    per-page scales (updated in place) covering every token ever appended to the page.
    k_min/k_max (optional) track the fp16 keys before quantisation."""
    k_new, v_new, slot_mapping = _check(k_new, v_new, k_cache, v_cache, slot_mapping)
    assert k_cache.dtype == torch.uint8, "FP8 pool holds e4m3 bits as uint8"
    num_blocks, block_size, Hkv, D = k_cache.shape
    assert block_size % KV_PAGE_TOKENS == 0, "FP8 pool needs block_size % 64 == 0"
    pages = num_blocks * (block_size // KV_PAGE_TOKENS)
    for s in (k_scale, v_scale):
        assert s.dtype == torch.float32 and s.is_contiguous() and s.numel() == pages * Hkv
    T = k_new.shape[0]
    if T == 0:
        return
    _maybe_minmax(k_new, slot_mapping, k_cache, k_min, k_max)

    # Sort the destination pages so each page's leader can be found without a host sync,
    # and snapshot the pre-append scales the existing bytes were encoded with.
    slots = slot_mapping.long()
    keys = torch.where(slots >= 0, slots // KV_PAGE_TOKENS, torch.full_like(slots, -1))
    keys, _ = torch.sort(keys)
    gather = keys.clamp_min(0)
    old_ks = k_scale.view(pages, Hkv)[gather].contiguous()
    old_vs = v_scale.view(pages, Hkv)[gather].contiguous()

    block_hkv = triton.next_power_of_2(Hkv)
    _kv_absmax_kernel[(T,)](k_new, v_new, slot_mapping, k_scale.view(torch.int32),
                            v_scale.view(torch.int32), Hkv,
                            D=D, PAGE=KV_PAGE_TOKENS, BLOCK_HKV=block_hkv)
    _kv_rescale_kernel[(T, Hkv)](k_cache, v_cache, k_scale, v_scale, keys, old_ks, old_vs, Hkv,
                                 D=D, PAGE=KV_PAGE_TOKENS)
    _kv_store_kernel[(T,)](k_new, v_new, k_cache, v_cache, k_scale, v_scale, slot_mapping, Hkv,
                           D=D, PAGE=KV_PAGE_TOKENS, BLOCK_HKV=block_hkv, FP8=True)


def apply_rope(x, cos, sin, positions):
    """NeoX-style rotary embedding. x [T, H, D]; cos/sin [max_pos, D // 2]; positions [T]."""
    half = x.shape[-1] // 2
    c = cos[positions].unsqueeze(1).float()
    s = sin[positions].unsqueeze(1).float()
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)


def apply_rope_and_cache(k_new, v_new, k_cache, v_cache, slot_mapping, cos, sin, positions,
                         k_scale=None, v_scale=None, k_min=None, k_max=None):
    """Rotate K, then append to the pool (FP8 when scales are given). Returns the rotated K.
    k_min/k_max (optional) are the Quest page bounds of the rotated keys."""
    k_rot = apply_rope(k_new, cos, sin, positions)
    if k_scale is None:
        reshape_and_cache(k_rot, v_new, k_cache, v_cache, slot_mapping, k_min, k_max)
    else:
        reshape_and_cache_fp8(k_rot, v_new, k_cache, v_cache, k_scale, v_scale, slot_mapping,
                              k_min, k_max)
    return k_rot
