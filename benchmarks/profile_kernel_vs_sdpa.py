"""
Isolated attention-kernel benchmark: custom split-KV `fused_gqa` vs PyTorch SDPA
(FlashAttention family), single decode step, Llama-3.1-8B attention shapes.

Why this exists:
  The full-model llama_integration_benchmark measures a whole model() decode, which
  at batch=1 is ~85% non-attention (eager overhead + weight GEMMs) and also pays a
  per-step KV transpose in the patched forward. That makes it the WRONG yardstick for
  the attention kernel. This script isolates the kernel — no model, no eager overhead,
  no transpose in the timed region — so we get an apples-to-apples kernel-vs-FA2 number,
  the way FlashDecoding / FlashInfer report.

Run (no model download, no HF login, tiny GPU footprint):
  python benchmarks/profile_kernel_vs_sdpa.py
  python benchmarks/profile_kernel_vs_sdpa.py --label stage1-bank-conflict

Every run is saved to results/kernel_vs_sdpa/ pinned to the git SHA, so the
optimisation stages form an auditable progression rather than a claim.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_kv._C as _C                     # noqa: E402
from bench_common import save_result          # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--label", default="",
                 help="names this run in results/ (e.g. stage1-bank-conflict)")
_ap.add_argument("--kernel", default="v3", choices=["v3", "v4"],
                 help="v3 = block per query head; v4 = block per KV head")
_args = _ap.parse_args()

if _args.kernel == "v4":
    if not hasattr(_C, "fused_gqa_v4"):
        raise SystemExit("fused_gqa_v4 not in the extension - rebuild: "
                         "python setup.py build_ext --inplace")
    fused_gqa = _C.fused_gqa_v4
else:
    fused_gqa = _C.fused_gqa

if hasattr(_C, "gqa_kernel_info"):
    _i = _C.gqa_kernel_info(32, 8)
    print("kernel occupancy (from the CUDA runtime, no profiler needed):")
    print("  v3: {:>3} regs, {:>6} B smem, {} blocks/SM x {} thr = {} thr/SM "
          "({:.1f}% of 2048)".format(
              _i[0], _i[1], _i[2], _i[3], _i[2] * _i[3], _i[2] * _i[3] / 20.48))
    print("  v4: {:>3} regs, {:>6} B smem, {} blocks/SM x {} thr = {} thr/SM "
          "({:.1f}% of 2048)".format(
              _i[4], _i[5], _i[6], _i[7], _i[6] * _i[7], _i[6] * _i[7] / 20.48))
    print("  SMs: {}\n".format(_i[8]))

torch.manual_seed(0)
dev = "cuda"

# Llama-3.1-8B attention config
Hq, Hkv, D = 32, 8, 128
B = 1
CTX = [4096, 16384, 64000]
scale = 1.0 / (D ** 0.5)
HBM_TBs = 3.35          # H100 SXM HBM3 peak bandwidth (TB/s)


def timed(fn, warmup=25, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


print(f"{'ctx':>8} {'v3(ms)':>10} {'SDPA(ms)':>10} {'v3/SDPA':>9} "
      f"{'v3 GB/s':>9} {'%HBM':>7} {'max_err':>9}")
print("-" * 72)

rows = []
for N in CTX:
    # custom-kernel layout: Q [B,Hq,D], K/V [B,N,Hkv,D]  (kernel does GQA internally)
    Q  = torch.randn(B, Hq, D,      device=dev, dtype=torch.float16)
    Kc = torch.randn(B, N, Hkv, D,  device=dev, dtype=torch.float16)
    Vc = torch.randn(B, N, Hkv, D,  device=dev, dtype=torch.float16)

    # SDPA layout: Q [B,Hq,1,D], K/V [B,Hkv,N,D]  (built ONCE, outside timing)
    Qs = Q.unsqueeze(2)
    Ks = Kc.transpose(1, 2).contiguous()
    Vs = Vc.transpose(1, 2).contiguous()

    # Prefer GQA-aware SDPA (torch>=2.5) so FA2 reads un-repeated KV — fair to both.
    # Fall back to manual head expansion on older torch.
    try:
        F.scaled_dot_product_attention(Qs, Ks, Vs, scale=scale, enable_gqa=True)
        sdpa_fn = lambda: F.scaled_dot_product_attention(
            Qs, Ks, Vs, scale=scale, enable_gqa=True)
    except TypeError:
        n = Hq // Hkv
        Ke = Ks[:, :, None].expand(B, Hkv, n, N, D).reshape(B, Hq, N, D).contiguous()
        Ve = Vs[:, :, None].expand(B, Hkv, n, N, D).reshape(B, Hq, N, D).contiguous()
        sdpa_fn = lambda: F.scaled_dot_product_attention(Qs, Ke, Ve, scale=scale)

    v3_fn = lambda: fused_gqa(Q, Kc, Vc, scale)

    # correctness (custom vs SDPA)
    o_v3  = fused_gqa(Q, Kc, Vc, scale)            # [B, Hq, D]
    o_ref = sdpa_fn().squeeze(2)                   # [B, Hq, D]
    max_err = (o_v3.float() - o_ref.float()).abs().max().item()

    v3_ms   = timed(v3_fn)
    sdpa_ms = timed(sdpa_fn)

    kv_bytes = 2 * N * Hkv * D * 2                 # K+V fp16, un-repeated
    gbps = kv_bytes / (v3_ms * 1e-3) / 1e9
    pct  = gbps / (HBM_TBs * 1000.0) * 100.0

    print(f"{N:>8} {v3_ms:>10.4f} {sdpa_ms:>10.4f} {v3_ms/sdpa_ms:>8.2f}x "
          f"{gbps:>9.0f} {pct:>6.1f}% {max_err:>9.5f}")

    rows.append({"ctx": N, "batch": B, "kernel": _args.kernel,
                 "ours_ms": v3_ms, "sdpa_ms": sdpa_ms,
                 "ratio": v3_ms / sdpa_ms,
                 "kv_gb": kv_bytes / 1e9,
                 "ours_gbps": gbps, "ours_pct_hbm": pct,
                 "max_err": max_err})

print("-" * 72)
print("v3/SDPA < 1.0 = our kernel is faster;  %HBM = fraction of peak bandwidth used")
print("(single decode step, batch=1, isolated — no model, no eager overhead)")

save_result("kernel_vs_sdpa", rows, label=_args.label,
            extra={"hq": Hq, "hkv": Hkv, "d": D, "hbm_tbs": HBM_TBs})
