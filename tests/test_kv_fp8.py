"""Quality gates for the FP8 (e4m3) KV cache.

FP8 is lossy by construction, so unlike the v4 tests these are not
bit-equality checks. e4m3 carries 3 mantissa bits, i.e. ~6% relative
resolution, recovered somewhat by per-page scaling and by attention being a
weighted average (independent errors partially cancel).

What must hold:
  - quantise/dequantise round-trips within the format's resolution
  - the decode output stays directionally identical to the fp16 path
    (cosine similarity), because that is what decides the sampled token
  - nothing degenerates to NaN/Inf

The end-to-end quality claim (argmax parity, perplexity delta on real text)
belongs in the benchmarks, not here -- this only gates the kernel.
"""
import pytest
import torch
import torch.nn.functional as F

import sparse_kv._C as _C

HQ, HKV, D = 32, 8, 128
SCALE = 1.0 / (D ** 0.5)
PAGE = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(_C, "fused_gqa_v4_fp8"),
    reason="needs CUDA and a build containing the fp8 kernels")


def _tensors(B, N, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, HQ, D, device="cuda", dtype=torch.float16),
            torch.randn(B, N, HKV, D, device="cuda", dtype=torch.float16),
            torch.randn(B, N, HKV, D, device="cuda", dtype=torch.float16))


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def test_quantize_shapes_and_pages():
    B, N = 1, 1000                       # deliberately not a page multiple
    _, K, V = _tensors(B, N)
    Kq, Vq, ks, vs = _C.quantize_kv_fp8(K, V)
    pages = (N + PAGE - 1) // PAGE
    assert Kq.shape == K.shape and Kq.dtype == torch.uint8
    assert Vq.shape == V.shape and Vq.dtype == torch.uint8
    assert ks.shape == (B, pages, HKV) and ks.dtype == torch.float32
    assert vs.shape == (B, pages, HKV)
    assert (ks > 0).all() and (vs > 0).all(), "a zero scale would divide by zero"


def test_quantize_halves_the_bytes():
    """The entire point: KV traffic and residency drop 2x."""
    _, K, V = _tensors(1, 4096)
    Kq, Vq, ks, vs = _C.quantize_kv_fp8(K, V)
    fp16 = K.numel() * K.element_size() + V.numel() * V.element_size()
    fp8 = (Kq.numel() + Vq.numel() + ks.numel() * 4 + vs.numel() * 4)
    assert fp8 < 0.55 * fp16, "expected ~2x reduction, got {:.2f}x".format(fp16 / fp8)


def test_zero_page_does_not_produce_nan():
    """An all-zero page has absmax 0; the scale must not become 0 or NaN."""
    Q = torch.randn(1, HQ, D, device="cuda", dtype=torch.float16)
    K = torch.zeros(1, 256, HKV, D, device="cuda", dtype=torch.float16)
    V = torch.zeros(1, 256, HKV, D, device="cuda", dtype=torch.float16)
    Kq, Vq, ks, vs = _C.quantize_kv_fp8(K, V)
    assert torch.isfinite(ks).all() and (ks > 0).all()
    out = _C.fused_gqa_v4_fp8(Q, Kq, Vq, ks, vs, SCALE)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("N", [64, 65, 512, 4096, 16384])
def test_fp8_close_to_fp16_path(N):
    """Directional agreement with the exact kernel across tile/page boundaries."""
    Q, K, V = _tensors(1, N)
    Kq, Vq, ks, vs = _C.quantize_kv_fp8(K, V)
    ref = _C.fused_gqa_v4(Q, K, V, SCALE)
    out = _C.fused_gqa_v4_fp8(Q, Kq, Vq, ks, vs, SCALE)
    assert torch.isfinite(out).all()
    cos = _cos(out, ref)
    assert cos > 0.99, "cosine {:.5f} vs fp16 path at N={}".format(cos, N)


@pytest.mark.parametrize("B", [1, 4])
def test_fp8_batched(B):
    Q, K, V = _tensors(B, 2048)
    Kq, Vq, ks, vs = _C.quantize_kv_fp8(K, V)
    ref = _C.fused_gqa_v4(Q, K, V, SCALE)
    out = _C.fused_gqa_v4_fp8(Q, Kq, Vq, ks, vs, SCALE)
    assert _cos(out, ref) > 0.99


def test_fp8_error_is_bounded_not_merely_finite():
    """Guards against a silently-wrong scale: report the actual relative error."""
    Q, K, V = _tensors(1, 8192)
    Kq, Vq, ks, vs = _C.quantize_kv_fp8(K, V)
    ref = _C.fused_gqa_v4(Q, K, V, SCALE).float()
    out = _C.fused_gqa_v4_fp8(Q, Kq, Vq, ks, vs, SCALE).float()
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 0.05, "relative L2 error {:.4f} exceeds e4m3 expectation".format(rel)
