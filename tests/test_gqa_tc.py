"""Correctness & zero-spill gate for the tensor-core + cp.async decode kernel (WP-3).

Tests fused_gqa_tc and fused_gqa_tc_fp8 across:
  - sequence lengths spanning 64-token tile/page boundaries: [1, 63, 64, 65, 1000, 16384]
  - batch sizes B in [1, 4]
  - speculative query tokens S_q in [1, 4] (both non-causal and causal verification)
  - head_dim D in [64, 128] and G = Hq // Hkv in [1, 4, 8] (including G*S_q > 16)
  - zero local-memory spills via gqa_tc_kernel_info
"""
import pytest
import torch
import torch.nn.functional as F

try:
    import sparse_kv._C as _C
except ImportError:
    _C = None

from triton_kernels.gqa_decode import KV_PAGE_TOKENS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or _C is None or not hasattr(_C, "fused_gqa_tc"),
    reason="needs CUDA and a build containing fused_gqa_tc",
)


def _dequant_fp8(xq, scale):
    """Dequantize [B, N, Hkv, D] uint8 e4m3 bits using [B, pages, Hkv] fp32 scales -> fp32."""
    B, N, Hkv, D = xq.shape
    pages = scale.shape[1]
    pad = pages * KV_PAGE_TOKENS - N
    xf = xq.view(torch.float8_e4m3fn).float()
    if pad:
        xf = F.pad(xf, (0, 0, 0, 0, 0, pad))
    xf = xf.view(B, pages, KV_PAGE_TOKENS, Hkv, D) * scale[:, :, None, :, None]
    return xf.view(B, pages * KV_PAGE_TOKENS, Hkv, D)[:, :N].contiguous()


def _sdpa_fp32_ref(q, k, v, scale, causal=False):
    """Exact fp32 SDPA reference supporting [B, Hq, D] and [B, S_q, Hq, D]."""
    is_3d = q.dim() == 3
    q4 = q.unsqueeze(1) if is_3d else q
    B, S_q, Hq, D = q4.shape
    N, Hkv = k.shape[1], k.shape[2]
    G = Hq // Hkv

    qf = q4.permute(0, 2, 1, 3).float()           # [B, Hq, S_q, D]
    kf = k.permute(0, 2, 1, 3).float()            # [B, Hkv, N, D]
    vf = v.permute(0, 2, 1, 3).float()
    if G != 1:
        kf = kf[:, :, None].expand(B, Hkv, G, N, D).reshape(B, Hq, N, D)
        vf = vf[:, :, None].expand(B, Hkv, G, N, D).reshape(B, Hq, N, D)

    attn_mask = None
    if causal:
        q_pos = torch.arange(N - S_q, N, device=q.device)[:, None]
        k_pos = torch.arange(N, device=q.device)[None, :]
        attn_mask = k_pos <= q_pos

    out = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=attn_mask, scale=scale)
    out = out.permute(0, 2, 1, 3).contiguous()    # [B, S_q, Hq, D]
    return out.squeeze(1) if is_3d else out


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("fp8", [False, True])
def test_tc_zero_register_spills(D, fp8):
    regs, local_bytes, smem_bytes, bps, threads = _C.gqa_tc_kernel_info(D, fp8)
    assert threads == 128
    assert bps >= 1
    assert smem_bytes > 0
    assert local_bytes == 0, f"register spill detected ({local_bytes} B) for D={D}, fp8={fp8}"


@pytest.mark.parametrize("N", [1, 63, 64, 65, 1000, 16384])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("S_q", [1, 4])
def test_tc_fp16_matches_fp32_sdpa(N, B, S_q):
    Hq, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    g = torch.Generator(device="cuda").manual_seed(100 + N + B * 17 + S_q * 31)
    q = torch.randn(B, S_q, Hq, D, device="cuda", dtype=torch.float16, generator=g)
    if S_q == 1:
        # Exercise the 3D [B, Hq, D] entry path on half the cases
        q_in = q.squeeze(1) if (N % 2 == 1) else q
    else:
        q_in = q
    k = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.float16, generator=g)

    out = _C.fused_gqa_tc(q_in, k, v, scale)
    ref = _sdpa_fp32_ref(q_in, k, v, scale)
    err = (out.float() - ref).abs().max().item()
    assert out.shape == q_in.shape
    assert torch.isfinite(out).all()
    assert err < 2e-3, f"FP16 max_abs_err={err:.5f} at B={B}, N={N}, S_q={S_q}"


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("G", [1, 4, 8])
@pytest.mark.parametrize("S_q", [1, 4])
def test_tc_fp16_head_dims_and_groups(D, G, S_q):
    B, N, Hkv = 2, 513, 4
    Hq = Hkv * G
    scale = D ** -0.5
    g = torch.Generator(device="cuda").manual_seed(200 + D + G * 7 + S_q * 13)
    q = torch.randn(B, S_q, Hq, D, device="cuda", dtype=torch.float16, generator=g)
    k = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.float16, generator=g)

    out = _C.fused_gqa_tc(q, k, v, scale)
    ref = _sdpa_fp32_ref(q, k, v, scale)
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-3, f"FP16 max_abs_err={err:.5f} at D={D}, G={G}, S_q={S_q}"


@pytest.mark.parametrize("N", [4, 64, 65, 513])
def test_tc_fp16_causal_speculative_verify(N):
    B, S_q, Hq, Hkv, D = 2, 4, 32, 8, 128
    scale = D ** -0.5
    g = torch.Generator(device="cuda").manual_seed(300 + N)
    q = torch.randn(B, S_q, Hq, D, device="cuda", dtype=torch.float16, generator=g)
    k = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.float16, generator=g)

    out = _C.fused_gqa_tc(q, k, v, scale, causal=True)
    ref = _sdpa_fp32_ref(q, k, v, scale, causal=True)
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-3, f"causal FP16 max_abs_err={err:.5f} at N={N}"


@pytest.mark.parametrize("N", [1, 63, 64, 65, 1000, 16384])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("S_q", [1, 4])
def test_tc_fp8_matches_dequant_fp32_sdpa(N, B, S_q):
    Hq, Hkv, D = 32, 8, 128
    scale = D ** -0.5
    g = torch.Generator(device="cuda").manual_seed(400 + N + B * 19 + S_q * 37)
    q = torch.randn(B, S_q, Hq, D, device="cuda", dtype=torch.float16, generator=g)
    q_in = q.squeeze(1) if S_q == 1 else q
    k = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.float16, generator=g)
    kq, vq, ks, vs = _C.quantize_kv_fp8(k, v)

    out = _C.fused_gqa_tc_fp8(q_in, kq, vq, ks, vs, scale)
    ref = _sdpa_fp32_ref(q_in, _dequant_fp8(kq, ks), _dequant_fp8(vq, vs), scale)
    err = (out.float() - ref).abs().max().item()
    assert torch.isfinite(out).all()
    assert err < 2e-3, f"FP8 dequant max_abs_err={err:.5f} at B={B}, N={N}, S_q={S_q}"


def test_tc_fp8_d64_and_causal():
    B, S_q, N, Hq, Hkv, D = 2, 4, 257, 16, 4, 64
    scale = D ** -0.5
    g = torch.Generator(device="cuda").manual_seed(501)
    q = torch.randn(B, S_q, Hq, D, device="cuda", dtype=torch.float16, generator=g)
    k = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn(B, N, Hkv, D, device="cuda", dtype=torch.float16, generator=g)
    kq, vq, ks, vs = _C.quantize_kv_fp8(k, v)

    out = _C.fused_gqa_tc_fp8(q, kq, vq, ks, vs, scale, causal=True)
    ref = _sdpa_fp32_ref(q, _dequant_fp8(kq, ks), _dequant_fp8(vq, vs), scale, causal=True)
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-3, f"FP8 causal D=64 max_abs_err={err:.5f}"


def test_tc_custom_op_registration():
    assert hasattr(torch.ops.sparse_kv, "fused_gqa_tc")
    assert hasattr(torch.ops.sparse_kv, "fused_gqa_tc_fp8")
