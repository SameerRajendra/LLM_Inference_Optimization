"""Decode-attention benchmark: Triton vs the hand-written CUDA v4 kernels vs torch SDPA.

    python benchmarks/bench_triton_decode.py [--peak-gbs 3352] [--batches 1,4]

Llama-3.1-8B shapes (32 Q / 8 KV heads, head_dim 128), one decode step. The CUDA
rows appear only if the extension is built (python setup.py build_ext --inplace).
Writes results/triton/<gpu>/decode.{csv,md}, stamped with commit and versions.

Timing: median of --iters calls, each with a cold L2 (KV caches in serving never
sit in cache) and the GPU parked on a spin kernel while the host enqueues the call,
so Triton's Python launch cost is not charged to it as GPU idle time.
"""
import argparse
import csv
import os
import re
import statistics
import sys

import torch
import torch.nn.functional as F
import triton

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench_common import provenance  # noqa: E402
from triton_kernels.gqa_decode import KV_PAGE_TOKENS, gqa_decode, quantize_kv_fp8  # noqa: E402

try:
    import sparse_kv._C as _C
except ImportError:
    _C = None

HQ, HKV, D = 32, 8, 128
CONTEXTS = [4096, 8192, 16384, 32768, 65536]
PEAK_GBS = {"H100 80GB HBM3": 3352}  # H100 SXM; other GPUs need --peak-gbs
WARMUP = 25
SPIN_CYCLES = 5_000_000  # a few ms: longer than the host needs to enqueue one call


def time_ms(fn, iters, flush):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush.zero_()
        torch.cuda._sleep(SPIN_CYCLES)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def traffic_bytes(B, N, kv_bytes):
    """Q + K + V (+ fp8 scales) + O: what the kernel must move, KV counted once."""
    q = B * HQ * D * 2
    scales = 2 * B * triton.cdiv(N, KV_PAGE_TOKENS) * HKV * 4 if kv_bytes == 1 else 0
    return 2 * q + 2 * B * N * HKV * D * kv_bytes + scales


def kernels(B, N):
    """name -> (callable, bytes per KV element). Inputs are built once, outside timing."""
    q = torch.randn(B, HQ, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, N, HKV, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, N, HKV, D, device="cuda", dtype=torch.float16)
    kq, vq, ks, vs = quantize_kv_fp8(k, v)  # same e4m3 bits feed the CUDA and Triton fp8 rows
    scale = D ** -0.5
    qs, k_t, v_t = q.unsqueeze(2), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

    out = {"sdpa": (lambda: F.scaled_dot_product_attention(
        qs, k_t, v_t, scale=scale, enable_gqa=True).squeeze(2), 2)}
    if _C is not None and hasattr(_C, "fused_gqa_v4_fp8"):
        out["cuda_v4"] = (lambda: _C.fused_gqa_v4(q, k, v, scale), 2)
        out["cuda_v4_fp8"] = (lambda: _C.fused_gqa_v4_fp8(q, kq, vq, ks, vs, scale), 1)
    out["triton_fp16"] = (lambda: gqa_decode(q, k, v), 2)
    out["triton_fp8"] = (lambda: gqa_decode(q, kq, vq, k_scale=ks, v_scale=vs), 1)
    return out


def markdown(rows, prov, peak, iters):
    lines = [
        "# Decode attention: Triton vs CUDA v4 vs SDPA",
        "",
        f"GPU `{prov['gpu']}` | commit `{prov['git_sha'] or 'unknown'}` ({prov['git_branch']}) | "
        f"torch {prov['torch']} | triton {triton.__version__} | CUDA {prov['cuda']} | {prov['utc']}",
        "",
        f"Hq={HQ} Hkv={HKV} D={D}. Median of {iters} calls, cold L2. "
        f"% of peak uses {peak} GB/s. Max err is vs fp16 SDPA (fp8 rows include quantisation error).",
    ]
    for B in sorted({r["batch"] for r in rows}):
        lines += ["", f"## Batch {B}", "",
                  "| ctx | kernel | ms | GB/s | % peak | max err |", "|---:|---|---:|---:|---:|---:|"]
        lines += [f"| {r['ctx']} | {r['kernel']} | {r['ms']:.4f} | {r['gbps']:.0f} | "
                  f"{r['pct_peak']:.1f} | {r['max_err']:.5f} |" for r in rows if r["batch"] == B]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-gbs", type=float, help="peak HBM GB/s (built-in for H100 SXM only)")
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    prov = provenance()
    peak = args.peak_gbs or next((v for name, v in PEAK_GBS.items() if name in prov["gpu"]), None)
    if peak is None:
        sys.exit(f"unknown peak bandwidth for {prov['gpu']}: pass --peak-gbs")

    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")  # larger than any L2
    rows = []
    for B in [int(b) for b in args.batches.split(",")]:
        for N in CONTEXTS:
            impls = kernels(B, N)
            ref = impls["sdpa"][0]().float()
            for name, (fn, kv_bytes) in impls.items():
                ms = time_ms(fn, args.iters, flush)
                gbps = traffic_bytes(B, N, kv_bytes) / (ms * 1e-3) / 1e9
                rows.append({"batch": B, "ctx": N, "kernel": name, "ms": ms, "gbps": gbps,
                             "pct_peak": 100 * gbps / peak,
                             "max_err": (fn().float() - ref).abs().max().item()})
                print(f"B={B} N={N:>6} {name:<12} {ms:.4f} ms {gbps:7.0f} GB/s {rows[-1]['pct_peak']:5.1f}%")
            del impls, ref
            torch.cuda.empty_cache()

    slug = re.sub(r"[^a-z0-9]+", "-", prov["gpu"].lower()).strip("-")
    out_dir = os.path.join(ROOT, "results", "triton", slug)
    os.makedirs(out_dir, exist_ok=True)
    stamp = {"gpu": prov["gpu"], "git_sha": prov["git_sha"], "torch": prov["torch"],
             "triton": triton.__version__}
    with open(os.path.join(out_dir, "decode.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + list(stamp))
        w.writeheader()
        w.writerows({**r, **stamp} for r in rows)
    table = markdown(rows, prov, peak, args.iters)
    with open(os.path.join(out_dir, "decode.md"), "w") as f:
        f.write(table)
    print("\n" + table)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
