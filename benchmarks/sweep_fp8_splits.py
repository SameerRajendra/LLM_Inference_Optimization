"""Split-count sweep for the paged Triton decode kernel: is FP8 parallelism-starved?

Hypothesis (from results/vs_flashinfer_tc/): paged FP16 decode runs at 82-90% of HBM peak, but
paged FP8 reaches only 1.6-2.4 TB/s, and its effective bandwidth climbs with the CTA count
(264 CTAs at B=1 -> 512 at B=32). The default heuristic sizes the split-KV grid to ~2 CTAs/SM
(num_splits = cdiv(2 * SMs, B * Hkv)); FP8 tiles carry half the bytes of FP16 tiles and are capped
at BLOCK_N <= 64 (one scale page per tile), so each CTA keeps fewer bytes in flight (Little's law).

This sweeps num_splits as multiples of that default (same shapes, same pool tensors), times each
point with CUDA-graph replay, and reports effective HBM bandwidth plus the autotuner's pick.
  - FP8 time falls as splits grow           -> memory-level parallelism is the limiter.
  - FP8 time flat while FP16 is at roofline -> per-token compute (padded M=16 rows, e4m3->fp16
                                               converts) is the limiter instead.

Triton's autotune key (tokens_bucket, FP8, BLOCK_SIZE, D, G) does not include num_splits, so the
autotune cache is cleared before every point: each split count gets its own tuned tile config.

Run:
  python benchmarks/sweep_fp8_splits.py
  python benchmarks/sweep_fp8_splits.py --batches 1,4 --contexts 65536 --mults 1,2,4,8

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
import os
import sys

import torch
import triton

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import save_result  # noqa: E402
from sparse_kv import ops as kv_ops  # noqa: E402
from triton_kernels import paged_gqa_decode as _paged  # noqa: E402
from triton_kernels.gqa_decode import MAX_SPLITS  # noqa: E402
from triton_kernels.paged_kv import allocate_paged_kv_cache  # noqa: E402

HBM_PEAK_GBS = 3350.0  # H100 SXM5 HBM3


def default_splits(B, hkv, max_seq_len, block_size, sms):
    """Mirror of the heuristic in triton_kernels/paged_gqa_decode.py (requested count)."""
    max_blocks = triton.cdiv(max_seq_len, block_size)
    ns = min(triton.cdiv(2 * sms, B * hkv), max_blocks)
    return max(1, min(ns, max_blocks, MAX_SPLITS))


def effective_splits(requested, max_seq_len, block_size):
    """The kernel rounds splits to whole blocks; this is the grid.y it actually launches."""
    max_blocks = triton.cdiv(max_seq_len, block_size)
    ns = max(1, min(requested, max_blocks, MAX_SPLITS))
    split_len = triton.cdiv(max_blocks, ns) * block_size
    return triton.cdiv(max_seq_len, split_len)


def graph_time_ms(fn, iters):
    graph, _ = kv_ops.capture_cuda_graph(fn, warmup_iters=3)
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        graph.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def best_config_str():
    cfg = getattr(_paged._paged_split_kernel, "best_config", None)
    if cfg is None:
        return "?"
    return "BN{} w{} s{}".format(cfg.kwargs.get("BLOCK_N"), cfg.num_warps, cfg.num_stages)


def make_pool(B, ctx, hq, hkv, d, bs, fp8, dev):
    g = torch.Generator(device=dev).manual_seed(B * 100003 + ctx)
    bps = triton.cdiv(ctx, bs)
    nb = B * bps
    q = torch.randn(B, hq, d, device=dev, generator=g).half()
    bt = torch.randperm(nb, device=dev, generator=g).view(B, bps).int()
    sl = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    kc, vc, ks, vs = allocate_paged_kv_cache(nb, bs, hkv, d, fp8, dev)
    for c in (kc, vc):
        x = torch.randn(c.shape, device=dev, generator=g, dtype=torch.float16)
        c.copy_(x.to(torch.float8_e4m3fn).view(torch.uint8) if fp8 else x)
        del x
    if fp8:
        ks.uniform_(0.01, 0.02, generator=g)
        vs.uniform_(0.01, 0.02, generator=g)
    return q, kc, vc, ks, vs, bt, sl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batches", default="1,4,16,32")
    ap.add_argument("--contexts", default="16384,65536")
    ap.add_argument("--mults", default="0.5,1,2,4,8",
                    help="num_splits multipliers of the default heuristic")
    ap.add_argument("--dtypes", default="fp16,fp8")
    ap.add_argument("--hq", type=int, default=32)   # Qwen3-8B attention shape
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--label", default="fp8-split-sweep")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    # 1.0 (the default heuristic) first: it is the reference row, and a duplicate effective split
    # count from another multiplier must not displace it.
    mults = sorted({float(x) for x in args.mults.split(",")} | {1.0}, key=lambda m: (m != 1.0, m))
    dtypes = [x.strip() for x in args.dtypes.split(",")]
    dev, bs = "cuda", args.block_size
    hq, hkv, d = args.hq, args.hkv, args.d
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print("GPU {} ({} SMs), Hq={} Hkv={} D={} block_size={}".format(
        torch.cuda.get_device_name(), sms, hq, hkv, d, bs))

    rows = []
    fp16_default_ms = {}
    hdr = "{:>5} {:>3} {:>6} {:>5} {:>7} {:>6} {:>9} {:>8} {:>6} {:>14} {:>9}".format(
        "dtype", "B", "ctx", "mult", "splits", "CTAs", "ms", "GB/s", "%peak", "autotune", "vs fp16")
    for ctx in contexts:
        for B in batches:
            base = default_splits(B, hkv, ctx, bs, sms)
            for dt in dtypes:
                fp8 = dt == "fp8"
                q, kc, vc, ks, vs, bt, sl = make_pool(B, ctx, hq, hkv, d, bs, fp8, dev)
                kv_bytes = 2 * B * ctx * hkv * d * (1 if fp8 else 2)
                if fp8:
                    kv_bytes += 2 * ks.numel() * 4
                ref = kv_ops.paged_gqa_decode(q, kc, vc, bt, sl, ctx, k_scale=ks, v_scale=vs,
                                              num_splits=base)
                seen = set()
                print("\n" + hdr)
                for m in mults:
                    req = max(1, int(round(base * m)))
                    eff = effective_splits(req, ctx, bs)
                    if eff in seen:
                        continue
                    seen.add(eff)
                    _paged._paged_split_kernel.cache.clear()  # re-tune for this split count

                    def fn(req=req):
                        return kv_ops.paged_gqa_decode(q, kc, vc, bt, sl, ctx, k_scale=ks,
                                                       v_scale=vs, num_splits=req)

                    ms = graph_time_ms(fn, args.iters)
                    err = (fn().float() - ref.float()).abs().max().item()
                    gbs = kv_bytes / ms / 1e6
                    row = {"dtype": dt, "batch": B, "ctx": ctx, "mult": m,
                           "num_splits_requested": req, "num_splits": eff,
                           "ctas": B * hkv * eff, "ms": ms, "gb_s": gbs,
                           "pct_peak": 100 * gbs / HBM_PEAK_GBS, "autotune": best_config_str(),
                           "max_err_vs_default_splits": err, "default_splits": base}
                    if not fp8 and m == 1.0:
                        fp16_default_ms[(B, ctx)] = ms
                    ref16 = fp16_default_ms.get((B, ctx))
                    if fp8 and ref16:
                        row["time_over_fp16_default"] = ms / ref16
                    rows.append(row)
                    print("{:>5} {:>3} {:>6} {:>5} {:>7} {:>6} {:>9.4f} {:>8.0f} {:>5.1f}% {:>14} {:>9}"
                          .format(dt, B, ctx, "{:g}x".format(m), eff, row["ctas"], ms, gbs,
                                  row["pct_peak"], row["autotune"],
                                  "{:.2f}x".format(row["time_over_fp16_default"])
                                  if "time_over_fp16_default" in row else "-"))
                    if err > 5e-3:
                        print("      WARNING: max_err {:.2e} vs default splits".format(err))
                del q, kc, vc, ks, vs, bt, sl, ref
                torch.cuda.empty_cache()

    print("\nvs fp16 = FP8 time / FP16 time at default splits; a bandwidth-bound FP8 kernel "
          "would sit near 0.50x.")
    best = {}
    for r in rows:
        k = (r["dtype"], r["batch"], r["ctx"])
        if k not in best or r["ms"] < best[k]["ms"]:
            best[k] = r
    print("\nBest split count per shape (speedup vs default heuristic):")
    for (dt, B, ctx), r in sorted(best.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        dflt = next((x for x in rows if (x["dtype"], x["batch"], x["ctx"]) == (dt, B, ctx)
                     and x["mult"] == 1.0), None)
        sp = dflt["ms"] / r["ms"] if dflt else float("nan")
        print("  {:>5} B={:<3} ctx={:<6} splits {:>3} -> {:>3}  {:.4f} ms  {:.0f} GB/s  ({:.2f}x)"
              .format(dt, B, ctx, dflt["num_splits"] if dflt else -1, r["num_splits"], r["ms"],
                      r["gb_s"], sp))

    if not args.no_save:
        save_result("fp8_split_sweep", rows, label=args.label,
                    extra={"hq": hq, "hkv": hkv, "d": d, "block_size": bs, "sms": sms,
                           "mults": mults, "iters": args.iters,
                           "gpu": torch.cuda.get_device_name()})


if __name__ == "__main__":
    main()
