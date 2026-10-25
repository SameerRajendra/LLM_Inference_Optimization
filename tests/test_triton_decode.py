"""Correctness gate for the Triton decode kernel (fp16 and FP8 KV).

Runs on a GPU, or on CPU under the Triton interpreter:
    TRITON_INTERPRET=1 python -m pytest tests/test_triton_decode.py
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triton_kernels.gqa_decode import KV_PAGE_TOKENS, gqa_decode, quantize_kv_fp8  # noqa: E402

HQ, HKV, D = 32, 8, 128
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Explicit split counts exercise the combine kernel; None is the GPU heuristic.
SPLITS = [1, 4] + ([None] if DEVICE == "cuda" else [])

pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" and not INTERPRET, reason="needs a GPU or TRITON_INTERPRET=1")


def _inputs(B, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    make = lambda *shape: torch.randn(*shape, generator=g).half().to(DEVICE)  # noqa: E731
    return make(B, HQ, D), make(B, N, HKV, D), make(B, N, HKV, D)


def _reference(q, k, v):
    """fp32 SDPA with native GQA; the only thing the kernel may differ by is fp16 rounding."""
    out = F.scaled_dot_product_attention(
        q.float().unsqueeze(2), k.float().transpose(1, 2), v.float().transpose(1, 2),
        enable_gqa=True)
    return out.squeeze(2)


def _dequantize(bits, scale):
    N = bits.shape[1]
    s = scale.repeat_interleave(KV_PAGE_TOKENS, dim=1)[:, :N, :, None]
    return bits.view(torch.float8_e4m3fn).float() * s


@pytest.mark.parametrize("num_splits", SPLITS)
@pytest.mark.parametrize("N", [1, 63, 64, 1000, 4097])
@pytest.mark.parametrize("B", [1, 4])
def test_fp16_matches_reference(B, N, num_splits):
    q, k, v = _inputs(B, N)
    out = gqa_decode(q, k, v, num_splits=num_splits)
    torch.testing.assert_close(out.float(), _reference(q, k, v), atol=2e-3, rtol=0)


@pytest.mark.skipif(INTERPRET, reason="64K context is too slow for the interpreter")
def test_fp16_long_context():
    q, k, v = _inputs(1, 65536)
    out = gqa_decode(q, k, v)
    torch.testing.assert_close(out.float(), _reference(q, k, v), atol=2e-3, rtol=0)


@pytest.mark.parametrize("num_splits", SPLITS)
@pytest.mark.parametrize("N", [1, 63, 64, 1000, 4097])
@pytest.mark.parametrize("B", [1, 4])
def test_fp8_matches_dequantized_reference(B, N, num_splits):
    """Isolates the kernel from quantisation error: the reference sees the same dequantised K/V."""
    q, k, v = _inputs(B, N)
    kq, vq, ks, vs = quantize_kv_fp8(k, v)
    out = gqa_decode(q, kq, vq, k_scale=ks, v_scale=vs, num_splits=num_splits)
    ref = _reference(q, _dequantize(kq, ks), _dequantize(vq, vs))
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=0)


@pytest.mark.skipif(INTERPRET, reason="64K context is too slow for the interpreter")
def test_fp8_long_context():
    q, k, v = _inputs(1, 65536)
    kq, vq, ks, vs = quantize_kv_fp8(k, v)
    out = gqa_decode(q, kq, vq, k_scale=ks, v_scale=vs)
    ref = _reference(q, _dequantize(kq, ks), _dequantize(vq, vs))
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=0)


def test_quantizer_matches_cuda_extension():
    """The torch quantizer agrees with kv_fp8.cu up to the CUDA build's fast-math division.

    setup.py compiles with --use_fast_math, so kv_fp8.cu's absmax/448 and 1/scale are approximate
    fp32 divisions (a few ulp off IEEE). That can flip the rounding of a value sitting on an e4m3
    rounding boundary, so a rare byte lands one code away. The benchmark is unaffected: it feeds
    the same bytes to the CUDA and Triton kernels.
    """
    C = pytest.importorskip("sparse_kv._C", reason="CUDA extension not built")
    if DEVICE != "cuda":
        pytest.skip("CUDA quantizer needs a GPU")
    _, k, v = _inputs(2, 1000)
    kq, vq, ks, vs = quantize_kv_fp8(k, v)
    ckq, cvq, cks, cvs = C.quantize_kv_fp8(k, v)
    for ours, theirs in ((ks, cks), (vs, cvs)):
        torch.testing.assert_close(ours, theirs, rtol=4 * 2.0 ** -23, atol=0)
    for ours, theirs in ((kq, ckq), (vq, cvq)):
        diff = (ours.int() - theirs.int()).abs()     # same sign => adjacent codes differ by 1
        mismatch = (diff > 0).float().mean().item()
        print(f"fp8 bytes differing from the CUDA quantizer: {mismatch:.4%}")
        assert diff.max().item() <= 1 and mismatch < 1e-3
