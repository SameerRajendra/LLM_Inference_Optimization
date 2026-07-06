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
"""
import torch
import torch.nn.functional as F
from sparse_kv._C import fused_gqa

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

print("-" * 72)
print("v3/SDPA < 1.0 = our kernel is faster;  %HBM = fraction of peak bandwidth used")
print("(single decode step, batch=1, isolated — no model, no eager overhead)")
