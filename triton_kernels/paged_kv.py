"""Paged KV cache: allocation and the fused quantize + scatter append (WP-1).

Pool contract (shared with paged_gqa_decode.py):
    k_cache, v_cache  [num_blocks, block_size, Hkv, D]  fp16, or uint8 holding e4m3 bits
    k_scale, v_scale  [num_blocks, block_size // KV_PAGE_TOKENS, Hkv] fp32 (FP8 only;
                      [num_blocks, Hkv] is accepted when block_size == KV_PAGE_TOKENS)
    slot_mapping      [num_tokens] int, slot = physical_block * block_size + offset; < 0 skips

FP8 scales: one per (64-token page, kv head), equal to the largest |x| / 448 ever appended to
that page. A scale of 0 marks a page that is empty (or all zero); allocate scales with zeros.
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


def reshape_and_cache(k_new, v_new, k_cache, v_cache, slot_mapping):
    """fp16 pool: scatter [T, Hkv, D] K/V into their slots."""
    k_new, v_new, slot_mapping = _check(k_new, v_new, k_cache, v_cache, slot_mapping)
    assert k_cache.dtype == torch.float16
    T, Hkv, D = k_new.shape
    if T == 0:
        return
    _kv_store_kernel[(T,)](k_new, v_new, k_cache, v_cache, None, None, slot_mapping, Hkv,
                           D=D, PAGE=KV_PAGE_TOKENS,
                           BLOCK_HKV=triton.next_power_of_2(Hkv), FP8=False)


def reshape_and_cache_fp8(k_new, v_new, k_cache, v_cache, k_scale, v_scale, slot_mapping):
    """FP8 pool: quantise [T, Hkv, D] fp16 K/V to e4m3, scatter to their slots, and keep the
    per-page scales (updated in place) covering every token ever appended to the page."""
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
                         k_scale=None, v_scale=None):
    """Rotate K, then append to the pool (FP8 when scales are given). Returns the rotated K."""
    k_rot = apply_rope(k_new, cos, sin, positions)
    if k_scale is None:
        reshape_and_cache(k_rot, v_new, k_cache, v_cache, slot_mapping)
    else:
        reshape_and_cache_fp8(k_rot, v_new, k_cache, v_cache, k_scale, v_scale, slot_mapping)
    return k_rot
