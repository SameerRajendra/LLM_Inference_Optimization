"""
Serving-regime decode benchmark: sweeps (batch x context) and reports the
metrics an inference operator actually pays for.

Why this exists
---------------
`profile_kernel_vs_sdpa.py` measures ONE decode step at batch=1 in isolation.
That is the right yardstick for "is my kernel efficient", but it is the wrong
regime for "does this help serve a model": at batch=1 the 16 GB of weights
dominate traffic and the KV cache is only ~35% of bytes moved, so no amount of
attention speedup moves the needle.

The KV cache becomes the bottleneck when batch x context is large -- which is
exactly where long-context serving lives, and where it hurts twice:

  capacity : KV pool size caps how many concurrent sequences fit on the GPU
  bandwidth: every decode step re-reads the whole KV cache of every sequence

At 64K context on one H100 with Llama-3.1-8B, 16 GB of weights leaves ~60 GB
of pool at 128 KiB/token -> only ~7 concurrent sequences, each step re-reading
~59 GB. That is the problem this project attacks.

Columns
-------
Measured: attention-kernel latency for the whole batch (one layer), the bytes
that implies, and achieved bandwidth.
Projected (marked "proj"): full decode step = measured attention x n_layers +
weight-read time, and the resulting tokens/sec. These are a MODEL, not a
measurement -- they assume weights stream at peak and ignore non-attention
kernel overhead, so treat them as an upper bound. End-to-end truth comes from
llama_integration_benchmark.py.

Run:
  python benchmarks/bench_serving_regime.py
  python benchmarks/bench_serving_regime.py --batches 1,8,32 --contexts 16384,64000
  python benchmarks/bench_serving_regime.py --json results/serving_regime.json

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_kv._C import fused_gqa      # noqa: E402
from bench_common import save_result    # noqa: E402

# ---- Llama-3.1-8B ------------------------------------------------------------
HQ, HKV, D = 32, 8, 128
N_LAYERS = 32
WEIGHT_GB = 16.0            # 8B params @ fp16
HBM_TBS = 3.35              # H100 SXM HBM3 peak
HBM_GB = 80.0
RESERVE_GB = 4.0            # activations, workspace, fragmentation


def timed(fn, warmup=10, iters=50):
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


def kv_bytes_per_token(dtype_bytes=2):
    """K and V, all layers, un-repeated (GQA stores only HKV heads)."""
    return 2 * N_LAYERS * HKV * D * dtype_bytes


def max_concurrent(ctx, dtype_bytes=2):
    pool = (HBM_GB - WEIGHT_GB - RESERVE_GB) * 1e9
    return int(pool // (ctx * kv_bytes_per_token(dtype_bytes)))


def run_config(B, N, scale, do_sdpa=True):
    dev = "cuda"
    Q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16)
    Kc = torch.randn(B, N, HKV, D, device=dev, dtype=torch.float16)
    Vc = torch.randn(B, N, HKV, D, device=dev, dtype=torch.float16)

    row = {"batch": B, "ctx": N}

    ours_ms = timed(lambda: fused_gqa(Q, Kc, Vc, scale))
    row["ours_ms"] = ours_ms

    if do_sdpa:
        Qs = Q.unsqueeze(2)
        Ks = Kc.transpose(1, 2).contiguous()
        Vs = Vc.transpose(1, 2).contiguous()
        try:
            F.scaled_dot_product_attention(Qs, Ks, Vs, scale=scale, enable_gqa=True)

            def sdpa_fn():
                return F.scaled_dot_product_attention(
                    Qs, Ks, Vs, scale=scale, enable_gqa=True)
        except TypeError:
            n = HQ // HKV
            Ke = Ks[:, :, None].expand(B, HKV, n, N, D).reshape(B, HQ, N, D).contiguous()
            Ve = Vs[:, :, None].expand(B, HKV, n, N, D).reshape(B, HQ, N, D).contiguous()

            def sdpa_fn():
                return F.scaled_dot_product_attention(Qs, Ke, Ve, scale=scale)

        o_ref = sdpa_fn().squeeze(2)
        o_ours = fused_gqa(Q, Kc, Vc, scale)
        row["max_err"] = (o_ours.float() - o_ref.float()).abs().max().item()
        row["sdpa_ms"] = timed(sdpa_fn)
        del Qs, Ks, Vs, o_ref, o_ours

    # Bytes this layer's attention must move for the whole batch.
    layer_bytes = 2 * B * N * HKV * D * 2
    row["layer_gb"] = layer_bytes / 1e9
    row["ours_gbps"] = layer_bytes / (ours_ms * 1e-3) / 1e9
    row["ours_pct_hbm"] = row["ours_gbps"] / (HBM_TBS * 1000.0) * 100.0
    if "sdpa_ms" in row:
        sdpa_gbps = layer_bytes / (row["sdpa_ms"] * 1e-3) / 1e9
        row["sdpa_pct_hbm"] = sdpa_gbps / (HBM_TBS * 1000.0) * 100.0

    # --- projected full decode step (MODEL, not measured) ---
    weight_ms = WEIGHT_GB / (HBM_TBS * 1000.0) * 1000.0     # GB / (GB/ms)
    for tag, ms in (("ours", ours_ms), ("sdpa", row.get("sdpa_ms"))):
        if ms is None:
            continue
        step_ms = ms * N_LAYERS + weight_ms
        row[tag + "_step_ms_proj"] = step_ms
        row[tag + "_tok_s_proj"] = B / (step_ms * 1e-3)

    del Q, Kc, Vc
    torch.cuda.empty_cache()
    return row


def fits(B, N, headroom=0.75):
    """Skip configs whose tensors would not fit (ours + SDPA copies)."""
    need = 2 * (2 * B * N * HKV * D * 2)      # our layout + SDPA transposed copy
    free, _ = torch.cuda.mem_get_info()
    return need < free * headroom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,4,8,16,32")
    ap.add_argument("--contexts", default="4096,16384,64000")
    ap.add_argument("--json", default="",
                    help="extra copy at an explicit path (results/ is always written)")
    ap.add_argument("--label", default="",
                    help="names this run in results/ (e.g. stage1-bank-conflict)")
    ap.add_argument("--no-sdpa", action="store_true")
    args = ap.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    scale = 1.0 / (D ** 0.5)
    torch.manual_seed(0)

    print("KV cache capacity on one H100 (Llama-3.1-8B, {:.0f} GB weights, "
          "{:.0f} GB reserved)".format(WEIGHT_GB, RESERVE_GB))
    print("{:>8} {:>14} {:>16} {:>16}".format(
        "ctx", "KV/seq (GB)", "max seqs fp16", "max seqs fp8"))
    print("-" * 58)
    for N in contexts:
        kv_seq = N * kv_bytes_per_token() / 1e9
        print("{:>8} {:>14.2f} {:>16} {:>16}".format(
            N, kv_seq, max_concurrent(N, 2), max_concurrent(N, 1)))
    print()

    print("Decode attention, measured per layer for the whole batch "
          "(proj = modelled full step)")
    hdr = ("{:>6} {:>8} {:>10} {:>10} {:>9} {:>8} {:>8} {:>11} {:>11} {:>9}")
    print(hdr.format("batch", "ctx", "ours(ms)", "SDPA(ms)", "ratio",
                     "GB/layer", "%HBM", "step_ms*", "tok/s*", "max_err"))
    print("-" * 108)

    rows = []
    for N in contexts:
        for B in batches:
            if not fits(B, N):
                print("{:>6} {:>8}   skipped - would not fit in free HBM".format(B, N))
                continue
            r = run_config(B, N, scale, do_sdpa=not args.no_sdpa)
            rows.append(r)
            ratio = (r["ours_ms"] / r["sdpa_ms"]) if "sdpa_ms" in r else float("nan")
            print(hdr.format(
                r["batch"], r["ctx"],
                "{:.4f}".format(r["ours_ms"]),
                "{:.4f}".format(r["sdpa_ms"]) if "sdpa_ms" in r else "-",
                "{:.2f}x".format(ratio),
                "{:.2f}".format(r["layer_gb"]),
                "{:.1f}%".format(r["ours_pct_hbm"]),
                "{:.1f}".format(r["ours_step_ms_proj"]),
                "{:.0f}".format(r["ours_tok_s_proj"]),
                "{:.5f}".format(r.get("max_err", float("nan")))))
    print("-" * 108)
    print("* projected: attention x {} layers + {:.2f} ms weight read at peak BW. "
          "Upper bound, not measured.".format(
              N_LAYERS, WEIGHT_GB / (HBM_TBS * 1000.0) * 1000.0))
    print("ratio < 1.0 = our kernel is faster than SDPA/FlashAttention.")

    cfg = {"hq": HQ, "hkv": HKV, "d": D, "n_layers": N_LAYERS,
           "weight_gb": WEIGHT_GB, "hbm_tbs": HBM_TBS,
           "hbm_gb": HBM_GB, "reserve_gb": RESERVE_GB,
           "capacity": {str(N): {"kv_gb_per_seq": N * kv_bytes_per_token() / 1e9,
                                 "max_seqs_fp16": max_concurrent(N, 2),
                                 "max_seqs_fp8": max_concurrent(N, 1)}
                        for N in contexts}}
    save_result("serving_regime", rows, label=args.label, extra=cfg)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"config": cfg, "rows": rows}, f, indent=2)
        print("wrote " + args.json)


if __name__ == "__main__":
    main()
