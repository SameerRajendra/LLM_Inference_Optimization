"""Portable Triton GQA decode: the tensor-core answer to the hand-written CUDA kernel.

csrc/kernels/gqa_decode.cu plateaus because QK and PV are scalar FMA chains
(instruction-issue bound). Here the G query heads that share one KV head are
packed into the M dimension of tl.dot, so each K/V tile is read once and fed to
the tensor cores. Layouts match the CUDA kernel: q [B, Hq, D], k/v [B, N, Hkv, D].
"""
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HEAD_DIM = 128
# Tokens per FP8 scale page. Must equal KV_PAGE_TOKENS in csrc/kernels/fp8_common.cuh.
KV_PAGE_TOKENS = 64
E4M3_MAX = 448.0
LOG2E = 1.4426950408889634
# The combine kernel holds every split of one row in a single tile.
MAX_SPLITS = 128


def _fp8_page_aligned(configs, named_args, **kwargs):
    """A tile wider than a page would straddle two scales, so drop those configs."""
    if not kwargs["FP8"]:
        return configs
    return [c for c in configs if KV_PAGE_TOKENS % c.kwargs["BLOCK_N"] == 0]


CONFIGS = [triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=s)
           for bn in (32, 64, 128) for w in (4, 8) for s in (2, 3)]
if os.environ.get("TRITON_INTERPRET") == "1":
    CONFIGS = CONFIGS[:1]  # the interpreter has no GPU to benchmark on


@triton.autotune(configs=CONFIGS, key=["tokens_bucket", "FP8"],
                 prune_configs_by={"early_config_prune": _fp8_page_aligned})
@triton.jit
def _decode_split_kernel(
    Q, K, V, KS, VS, O, M, L,
    N, split_len, num_splits, num_pages, qk_scale,
    tokens_bucket,  # autotune key only: batch * context, rounded to a power of 2
    Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, FP8: tl.constexpr,
):
    """One program = (batch, kv head, KV split); all G query heads share each K/V tile."""
    pid, split = tl.program_id(0), tl.program_id(1)
    b = (pid // Hkv).to(tl.int64)
    h = pid % Hkv
    Hq = Hkv * G
    rows = tl.arange(0, BLOCK_H)
    cols = tl.arange(0, D)
    row_ok = rows < G   # BLOCK_H >= 16 is tl.dot's minimum M; the extra rows are padding

    q = tl.load(Q + ((b * Hq + h * G + rows) * D)[:, None] + cols[None, :],
                mask=row_ok[:, None], other=0.0)

    start = split * split_len
    end = tl.minimum(start + split_len, N)
    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)  # base-2 domain, see qk_scale
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)

    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        in_range = offs_n < end
        tile = (((b * N + offs_n) * Hkv + h) * D)[:, None] + cols[None, :]
        k = tl.load(K + tile, mask=in_range[:, None], other=0)
        v = tl.load(V + tile, mask=in_range[:, None], other=0)
        if FP8:
            # e4m3 bits -> fp16 is exact. The page scale is constant over the tile,
            # so it factors out of both dots and is applied to the small results.
            k = k.to(tl.float8e4nv, bitcast=True)
            v = v.to(tl.float8e4nv, bitcast=True)
            sc = (b * num_pages + n0 // PAGE) * Hkv + h
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
def _combine_kernel(O, M, L, Out, num_splits, D: tl.constexpr, BLOCK_S: tl.constexpr):
    """Exact log-sum-exp merge of the per-split partials of one (batch, q head) row."""
    row = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, D)
    ok = s < num_splits
    m = tl.load(M + row * num_splits + s, mask=ok, other=float("-inf"))
    denom = tl.load(L + row * num_splits + s, mask=ok, other=0.0)
    o = tl.load(O + ((row * num_splits + s) * D)[:, None] + d[None, :],
                mask=ok[:, None], other=0.0)
    w = tl.exp2(m - tl.max(m, 0))
    out = tl.sum(o * w[:, None], 0) / tl.sum(denom * w, 0)
    tl.store(Out + row * D + d, out.to(tl.float16))


def gqa_decode(q, k, v, scale=None, k_scale=None, v_scale=None, num_splits=None):
    """Decode attention for one new token. Passing k_scale/v_scale selects the FP8 path.

    q [B, Hq, D] fp16; k, v [B, N, Hkv, D] fp16, or uint8 e4m3 bits with scales
    [B, ceil(N / KV_PAGE_TOKENS), Hkv] fp32 (as produced by quantize_kv_fp8).
    """
    B, Hq, D = q.shape
    _, N, Hkv, _ = k.shape
    fp8 = k_scale is not None
    assert D == HEAD_DIM and Hq % Hkv == 0, "needs head_dim 128 and Hq % Hkv == 0"
    assert k.dtype == (torch.uint8 if fp8 else torch.float16)
    G = Hq // Hkv
    num_pages = triton.cdiv(N, KV_PAGE_TOKENS)

    if num_splits is None:
        # Aim for ~2 programs per SM, but never split finer than one page.
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count if q.is_cuda else 1
        num_splits = min(triton.cdiv(2 * sms, B * Hkv), num_pages)
    num_splits = min(num_splits, MAX_SPLITS)
    # Page-aligned splits keep every FP8 tile inside one scale page.
    split_len = triton.cdiv(triton.cdiv(N, num_splits), KV_PAGE_TOKENS) * KV_PAGE_TOKENS
    num_splits = triton.cdiv(N, split_len)

    f32 = dict(dtype=torch.float32, device=q.device)
    O = torch.empty(B, Hq, num_splits, D, **f32)
    M = torch.empty(B, Hq, num_splits, **f32)
    L = torch.empty(B, Hq, num_splits, **f32)
    out = torch.empty_like(q)

    scale = D ** -0.5 if scale is None else scale
    _decode_split_kernel[(B * Hkv, num_splits)](
        q, k, v, k_scale, v_scale, O, M, L,
        N, split_len, num_splits, num_pages, scale * LOG2E,
        triton.next_power_of_2(B * N),
        Hkv=Hkv, G=G, D=D, PAGE=KV_PAGE_TOKENS,
        BLOCK_H=max(16, triton.next_power_of_2(G)), FP8=fp8)
    _combine_kernel[(B * Hq,)](O, M, L, out, num_splits, D=D,
                               BLOCK_S=triton.next_power_of_2(num_splits))
    return out


def quantize_kv_fp8(k, v):
    """Torch mirror of csrc/kernels/kv_fp8.cu: e4m3 bits (uint8) plus one fp32 scale
    per (page, kv head), scale = page absmax / 448 (1 for an all-zero page).
    Lets the Triton path run, and be tested, without the CUDA extension."""
    def quantize(x):
        B, N, H, D = x.shape
        pages = triton.cdiv(N, KV_PAGE_TOKENS)
        padded = F.pad(x, (0, 0, 0, 0, 0, pages * KV_PAGE_TOKENS - N))
        absmax = padded.view(B, pages, KV_PAGE_TOKENS, H, D).abs().amax(dim=(2, 4)).float()
        scale = torch.where(absmax > 0, absmax / E4M3_MAX, torch.ones_like(absmax))
        inv = (1.0 / scale).repeat_interleave(KV_PAGE_TOKENS, dim=1)[:, :N, :, None]
        bits = (x.float() * inv).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return bits.view(torch.uint8), scale
    (kq, ks), (vq, vs) = quantize(k), quantize(v)
    return kq, vq, ks, vs
