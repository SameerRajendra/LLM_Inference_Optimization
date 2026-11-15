"""Head-to-head vs FlashInfer and vLLM at matched (batch, context) on one GPU (WP-2).

Engines (each optional; a missing package skips its rows instead of failing the run):
  sparse_kv   our engine (bench_e2e_serving.py): paged fp16 / fp8 KV, whole decode step in one
              CUDA graph. Metric: TPOT = mean per-step latency from CUDA events.
  vllm        vllm.LLM with the same model, kv_cache_dtype auto (fp16 model) or fp8 (bf16 model,
              required by Hopper FA3 FP8), CUDA graphs on, prefix caching off, random-token prompts
              of length ctx.
              Metric: TPOT = (t(steps + 1 tokens) - t(1 token)) / steps, wall clock, fresh prompts
              each call. That includes vLLM's scheduler / sampler per step: an engine-level
              number, as is ours.
  flashinfer  kernel-level: one layer of paged decode attention (BatchDecodeWithPagedKVCache,
              NHD layout, the same pool tensors) vs torch.ops.sparse_kv.paged_gqa_decode, both
              replayed from CUDA graphs. FlashInfer FP8 uses a per-tensor scale (ours is
              per-page), so FP8 rows are timing-only.
              --flashinfer-tensor-cores picks FlashInfer's decode path: `off` (the wrapper
              default, use_tensor_cores=False -> CUDA-core "fa2" decode kernel), `on`
              (use_tensor_cores=True -> tensor-core prefill-style kernel, which FlashInfer
              documents as faster for large GQA groups), or `both`. Rows of the `on` path get
              method kernel_fp16_tc / kernel_fp8_tc; `off` keeps kernel_fp16 / kernel_fp8.

`--engine` takes one engine or a comma-separated list (`all` = sparse_kv,flashinfer,vllm). With
more than one, every engine runs in its own subprocess (vLLM reserves most of the GPU, and its
engine does not tear down cleanly in-process; vllm runs once per kv dtype, auto and fp8), the
rows are merged and one result file is saved.

bench_e2e_serving.py already measures sparse_kv paged_fp16 / paged_fp8. To avoid running them
twice, leave sparse_kv out (or pass --skip-sparse-kv) and point --sparse-kv-results at that run's
results/e2e_serving/*.json: its paged rows are copied in unchanged (tagged with the source file)
so the summary still lines them up against vLLM.

Run:
  python benchmarks/bench_vs_engines.py --engine all
  python benchmarks/bench_vs_engines.py --engine vllm,flashinfer \
      --sparse-kv-results results/e2e_serving/<run>.json
  python benchmarks/bench_vs_engines.py --engine flashinfer --batches 1,16 --contexts 16384
  python benchmarks/bench_vs_engines.py --engine flashinfer --flashinfer-tensor-cores both \
      --bench-name vs_flashinfer_tc

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import save_result  # noqa: E402

try:
    import flashinfer
except ImportError:
    flashinfer = None
try:
    import vllm
except ImportError:
    vllm = None


# ---------------------------------------------------------------------------------- sparse_kv

def run_sparse_kv(args):
    from bench_e2e_serving import Model, load_model, run_matrix
    max_len = max(args.contexts) + args.warmup + args.steps
    hf, overrides = load_model(args.model, max_len)
    M = Model(hf, max_len)
    rows = run_matrix(M, args.model, args.batches, args.contexts, ["paged_fp16", "paged_fp8"],
                      args.steps, args.warmup, args.block_size)
    for r in rows:
        r["engine"] = "sparse_kv"
        if "itl_ms_mean" in r:
            r["tpot_ms"] = r["itl_ms_mean"]
            r["tok_s_from_tpot"] = r["batch"] / r["tpot_ms"] * 1e3
    return rows, {"sparse_kv_rope_overrides": overrides}


# --------------------------------------------------------------------------------------- vLLM

def run_vllm(args):
    if vllm is None:
        print("vllm not installed: skipping")
        return [], {}
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig

    from bench_e2e_serving import long_context_overrides
    max_len = max(args.contexts) + args.steps + 8
    overrides = {} if args.model == "tiny-random" else long_context_overrides(
        AutoConfig.from_pretrained(args.model), max_len)
    vllm_dtype = "bfloat16" if args.vllm_kv_dtype == "fp8" else "float16"
    kw = dict(model=args.model, dtype=vllm_dtype, kv_cache_dtype=args.vllm_kv_dtype,
              max_model_len=max_len, max_num_seqs=max(args.batches),
              gpu_memory_utilization=args.vllm_gpu_util, enable_prefix_caching=False,
              seed=0)
    if overrides:
        kw["hf_overrides"] = overrides
    llm = LLM(**kw)
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(0)
    method = "vllm_fp8" if args.vllm_kv_dtype == "fp8" else "vllm_fp16"

    def prompts(B, ctx):
        return [{"prompt_token_ids": [rng.randrange(vocab) for _ in range(ctx)]}
                for _ in range(B)]

    def timed(B, ctx, n):
        sp = SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.0,
                            detokenize=False)
        p = prompts(B, ctx)
        t0 = time.perf_counter()
        outs = llm.generate(p, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        assert all(len(o.outputs[0].token_ids) == n for o in outs)
        return dt

    timed(1, min(args.contexts), 4)  # first-call warm-up
    rows = []
    for ctx in args.contexts:
        for B in args.batches:
            t1 = timed(B, ctx, 1)
            tn = timed(B, ctx, args.steps + 1)
            tpot = (tn - t1) / args.steps * 1e3
            rows.append({"engine": "vllm", "method": method, "model_dtype": vllm_dtype,
                         "batch": B, "ctx": ctx, "steps": args.steps, "tpot_ms": tpot,
                         "tok_s_from_tpot": B / tpot * 1e3, "t_1tok_s": t1, "t_ntok_s": tn,
                         "model": args.model})
            print("{:>10} ({}) B={:<3} ctx={:<6} TPOT {:8.3f} ms  {:9.1f} tok/s".format(
                method, vllm_dtype, B, ctx, tpot, B / tpot * 1e3))
    return rows, {"vllm_version": vllm.__version__, "vllm_rope_overrides": overrides,
                  "vllm_gpu_memory_utilization": args.vllm_gpu_util,
                  "vllm_model_dtype_" + args.vllm_kv_dtype: vllm_dtype}


# --------------------------------------------------------------------------------- FlashInfer

def _graph_time(fn, iters):
    """Mean ms per call of fn replayed from a CUDA graph (falls back to eager launches)."""
    from sparse_kv import ops as kv_ops
    try:
        graph, _ = kv_ops.capture_cuda_graph(fn, warmup_iters=3)
        run, mode = graph.replay, "cuda_graph"
    except Exception as e:  # noqa: BLE001  (a backend that cannot be captured)
        print("  graph capture failed ({}); timing eager launches".format(e))
        torch.cuda.synchronize()
        run, mode = fn, "eager"
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        run()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, mode


def _attn_dims(model):
    if model == "tiny-random":
        return 8, 2, 128
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model)
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return cfg.num_attention_heads, cfg.num_key_value_heads, d


def run_flashinfer(args):
    if flashinfer is None:
        print("flashinfer not installed: skipping")
        return [], {}
    hq, hkv, d = _attn_dims(args.model)
    bs, dev = args.block_size, "cuda"
    tc_modes = {"off": [False], "on": [True], "both": [False, True]}[args.flashinfer_tensor_cores]
    rows = []
    for ctx in args.contexts:
        for B in args.batches:
            bps = -(-ctx // bs)
            nb = B * bps
            need = 2 * nb * bs * hkv * d * 2 * 2  # fp16 pool + a same-size fp8 view headroom
            if need > 0.8 * torch.cuda.mem_get_info()[0]:
                rows.append({"engine": "flashinfer", "batch": B, "ctx": ctx,
                             "skipped": "does not fit"})
                continue
            g = torch.Generator(device=dev).manual_seed(B * 100003 + ctx)
            q = torch.randn(B, hq, d, device=dev, generator=g).half()
            bt = torch.randperm(nb, device=dev, generator=g).view(B, bps).int()
            sl = torch.full((B,), ctx, dtype=torch.int32, device=dev)
            indptr = torch.arange(0, (B + 1) * bps, bps, dtype=torch.int32, device=dev)
            indices = bt.reshape(-1).contiguous()
            last = torch.full((B,), (ctx - 1) % bs + 1, dtype=torch.int32, device=dev)
            ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
            for fp8 in (False, True):
                case_rows = _flashinfer_case(fp8, q, bt, sl, indptr, indices, last, ws, g, ctx,
                                             (hq, hkv, d, bs), args.kernel_iters, tc_modes)
                for row in case_rows:
                    row.update({"engine": "flashinfer", "batch": B, "ctx": ctx})
                    rows.append(row)
                    if "flashinfer_error" in row:
                        print("{:>15} B={:<3} ctx={:<6} ours {:.4f} ms  flashinfer FAILED: {}"
                              .format(row["method"], B, ctx, row["ours_ms"],
                                      row["flashinfer_error"]))
                        continue
                    print("{:>15} B={:<3} ctx={:<6} ours {:.4f} ms  flashinfer {:.4f} ms  "
                          "({:.2f}x) [{}]{}".format(
                              row["method"], B, ctx, row["ours_ms"], row["flashinfer_ms"],
                              row["ours_over_flashinfer"], row["flashinfer_backend"],
                              "  max_err {:.2e}".format(row["max_err_vs_flashinfer"])
                              if "max_err_vs_flashinfer" in row else ""))
                gc.collect()
                torch.cuda.empty_cache()
    return rows, {"flashinfer_version": getattr(flashinfer, "__version__", "?"),
                  "flashinfer_tensor_cores": args.flashinfer_tensor_cores}


def _flashinfer_case(fp8, q, bt, sl, indptr, indices, last, ws, g, ctx, dims, iters, tc_modes):
    """One pool (fp16 or fp8) filled with random K/V; ours vs FlashInfer on the same tensors.

    Ours is timed once per pool; FlashInfer once per entry of tc_modes (use_tensor_cores
    False / True). Returns one row per FlashInfer config.
    """
    from sparse_kv import ops as kv_ops
    from triton_kernels.paged_kv import allocate_paged_kv_cache
    hq, hkv, d, bs = dims
    kc, vc, ks, vs = allocate_paged_kv_cache(bt.numel(), bs, hkv, d, fp8, q.device)
    for c in (kc, vc):
        x = torch.randn(c.shape, device=q.device, generator=g, dtype=torch.float16)
        c.copy_(x.to(torch.float8_e4m3fn).view(torch.uint8) if fp8 else x)
        del x
    if fp8:
        ks.uniform_(0.01, 0.02, generator=g)
        vs.uniform_(0.01, 0.02, generator=g)

    def ours():
        return kv_ops.paged_gqa_decode(q, kc, vc, bt, sl, ctx, k_scale=ks, v_scale=vs)

    ours_ms, ours_timing = _graph_time(ours, iters)

    kv_dtype = torch.float8_e4m3fn if fp8 else torch.float16
    fi_kv = (kc.view(kv_dtype), vc.view(kv_dtype)) if fp8 else (kc, vc)
    fi_kw = {"k_scale": 0.015, "v_scale": 0.015} if fp8 else {}

    rows = []
    for use_tc in tc_modes:
        row = {"method": "kernel_" + ("fp8" if fp8 else "fp16") + ("_tc" if use_tc else ""),
               "hq": hq, "hkv": hkv, "d": d, "block_size": bs,
               "flashinfer_use_tensor_cores": use_tc,
               "ours_ms": ours_ms, "ours_timing": ours_timing}
        try:
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                ws, "NHD", use_tensor_cores=use_tc)
            wrapper.plan(indptr, indices, last, hq, hkv, d, bs, pos_encoding_mode="NONE",
                         q_data_type=torch.float16, kv_data_type=kv_dtype)

            def theirs():
                return wrapper.run(q, fi_kv, **fi_kw)

            # What "auto" actually ran: "fa2" with use_tensor_cores=False is the CUDA-core
            # decode kernel; with True it is the tensor-core pick (fa2 / fa3).
            row["flashinfer_backend"] = str(getattr(wrapper, "resolved_backend", None)
                                            or getattr(wrapper, "_backend", "?"))
            if not fp8:  # same numerics: must agree. FP8 scales differ (per-page vs per-tensor).
                row["max_err_vs_flashinfer"] = (
                    ours().float() - theirs().float()).abs().max().item()
            row["flashinfer_ms"], row["flashinfer_timing"] = _graph_time(theirs, iters)
            row["ours_over_flashinfer"] = ours_ms / row["flashinfer_ms"]
        except Exception as e:  # noqa: BLE001  (e.g. a dtype combo the TC path rejects)
            row["flashinfer_error"] = "{}: {}".format(type(e).__name__, str(e)[:300])
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------- orchestration

ENGINES = {"sparse_kv": run_sparse_kv, "vllm": run_vllm, "flashinfer": run_flashinfer}
KV_SIZE_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")


def _child(args, engine, extra_flags):
    """Run one engine in a subprocess, streaming its output; returns (rows, extra, log)."""
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    cmd = [sys.executable, os.path.abspath(__file__), "--engine", engine, "--rows-out", out,
           "--model", args.model, "--batches", ",".join(map(str, args.batches)),
           "--contexts", ",".join(map(str, args.contexts)), "--steps", str(args.steps),
           "--warmup", str(args.warmup), "--block-size", str(args.block_size),
           "--kernel-iters", str(args.kernel_iters),
           "--flashinfer-tensor-cores", args.flashinfer_tensor_cores,
           "--vllm-gpu-util", str(args.vllm_gpu_util)] + extra_flags
    print("\n=== " + " ".join(cmd[2:4] + extra_flags))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=ROOT)
    log = []
    for line in proc.stdout:
        sys.stdout.write(line)
        log.append(line)
    proc.wait()
    if proc.returncode != 0 or os.path.getsize(out) == 0:
        print("engine {} {} failed (exit {})".format(engine, extra_flags, proc.returncode))
        os.remove(out)
        return [], {"{}{}_error".format(engine, "".join(extra_flags)): "exit {}".format(
            proc.returncode)}, "".join(log)
    with open(out) as f:
        payload = json.load(f)
    os.remove(out)
    return payload["rows"], payload["extra"], "".join(log)


def load_sparse_kv_results(path):
    """paged_fp16 / paged_fp8 rows of a bench_e2e_serving.py result file, as sparse_kv rows."""
    with open(path) as f:
        payload = json.load(f)
    rows = []
    for r in payload["rows"]:
        if r.get("method") not in ("paged_fp16", "paged_fp8"):
            continue
        r = dict(r, engine="sparse_kv", source=os.path.basename(path))
        if "itl_ms_mean" in r:
            r["tpot_ms"] = r["itl_ms_mean"]
            r["tok_s_from_tpot"] = r["batch"] / r["tpot_ms"] * 1e3
        rows.append(r)
    return rows, {"sparse_kv_source": os.path.basename(path),
                  "sparse_kv_source_provenance": payload.get("provenance")}


def run_jobs(args, engines):
    rows, extra = [], {}
    jobs = []
    for engine in engines:
        if engine == "vllm":
            jobs += [("vllm", ["--vllm-kv-dtype", "auto"]), ("vllm", ["--vllm-kv-dtype", "fp8"])]
        else:
            jobs.append((engine, []))
    for engine, flags in jobs:
        r, x, log = _child(args, engine, flags)
        if engine == "vllm":
            # A batch whose KV does not fit in vLLM's pool gets queued / preempted, and its TPOT
            # is no longer a decode-step latency: flag it from vLLM's own capacity log line.
            m = KV_SIZE_RE.findall(log)
            if m:
                cap = int(m[-1].replace(",", ""))
                x["vllm_kv_tokens_" + flags[-1]] = cap
                for row in r:
                    row["vllm_kv_tokens"] = cap
                    row["fits_in_vllm_kv"] = row["batch"] * (row["ctx"] + args.steps + 1) <= cap
        rows += r
        extra.update(x)
    return rows, extra


def summarize(rows):
    def get(engine, method, B, ctx, key):
        for r in rows:
            if (r.get("engine"), r.get("method"), r.get("batch"), r.get("ctx")) == (
                    engine, method, B, ctx) and key in r:
                return r[key]
        return None

    keys = sorted({(r["ctx"], r["batch"]) for r in rows if "batch" in r})
    has_tc = any(str(r.get("method", "")).endswith("_tc") for r in rows)
    print("\nTPOT ms (lower is better)        ours fp16  vllm fp16   ours fp8   vllm fp8"
          "   | attn kernel ours/flashinfer:     fp16      fp8"
          + ("   fp16 TC    fp8 TC" if has_tc else ""))
    fmt = lambda x: "{:10.3f}".format(x) if x is not None else "{:>10}".format("-")  # noqa: E731
    rfmt = lambda x: "{:8.2f}x".format(x) if x is not None else "{:>9}".format("-")  # noqa: E731
    for ctx, B in keys:
        tc_cols = (" " + rfmt(get("flashinfer", "kernel_fp16_tc", B, ctx, "ours_over_flashinfer"))
                   + " " + rfmt(get("flashinfer", "kernel_fp8_tc", B, ctx,
                                    "ours_over_flashinfer"))) if has_tc else ""
        print("ctx {:>6} B {:>3}               {} {} {} {}   |                         {} {}{}"
              .format(ctx, B, fmt(get("sparse_kv", "paged_fp16", B, ctx, "tpot_ms")),
                      fmt(get("vllm", "vllm_fp16", B, ctx, "tpot_ms")),
                      fmt(get("sparse_kv", "paged_fp8", B, ctx, "tpot_ms")),
                      fmt(get("vllm", "vllm_fp8", B, ctx, "tpot_ms")),
                      rfmt(get("flashinfer", "kernel_fp16", B, ctx, "ours_over_flashinfer")),
                      rfmt(get("flashinfer", "kernel_fp8", B, ctx, "ours_over_flashinfer")),
                      tc_cols))
    print("kernel ratio < 1.0 = ours faster; fp8 kernel rows are timing-only (scales differ)."
          + (" TC = FlashInfer use_tensor_cores=True." if has_tc else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", default="all",
                    help="all, or a comma-separated subset of: " + ",".join(sorted(ENGINES)))
    ap.add_argument("--skip-sparse-kv", action="store_true",
                    help="drop sparse_kv (already measured by bench_e2e_serving.py)")
    ap.add_argument("--sparse-kv-results", default="",
                    help="bench_e2e_serving.py result JSON whose paged rows join the summary "
                         "(implies --skip-sparse-kv)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batches", default="1,4,8,16,32")
    ap.add_argument("--contexts", default="4096,16384,65536")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--kernel-iters", type=int, default=100)
    ap.add_argument("--flashinfer-tensor-cores", default="off", choices=["off", "on", "both"],
                    help="FlashInfer decode path: off = wrapper default (CUDA-core fa2 decode), "
                         "on = use_tensor_cores=True, both = time each (rows *_tc for on)")
    ap.add_argument("--bench-name", default="vs_engines",
                    help="results/<bench-name>/ directory to save into (the report scripts read "
                         "the newest results/vs_engines run, so save side runs elsewhere)")
    ap.add_argument("--vllm-kv-dtype", default="auto", choices=["auto", "fp8"])
    ap.add_argument("--vllm-gpu-util", type=float, default=0.90)
    ap.add_argument("--rows-out", default="", help=argparse.SUPPRESS)  # subprocess hand-off
    ap.add_argument("--label", default="")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    args.batches = [int(x) for x in args.batches.split(",")]
    args.contexts = [int(x) for x in args.contexts.split(",")]

    engines = sorted(ENGINES) if args.engine == "all" else [
        e.strip() for e in args.engine.split(",") if e.strip()]
    for e in engines:
        if e not in ENGINES:
            raise SystemExit("unknown engine {} (choose from {})".format(e, sorted(ENGINES)))
    if args.skip_sparse_kv or args.sparse_kv_results:
        engines = [e for e in engines if e != "sparse_kv"]

    # One engine runs in-process (always so in a child); several, or vllm at both kv dtypes,
    # run as isolated subprocesses.
    if len(engines) == 1 and (engines[0] != "vllm" or args.rows_out):
        rows, extra = ENGINES[engines[0]](args)
    else:
        rows, extra = run_jobs(args, engines)
    if args.sparse_kv_results:
        r, x = load_sparse_kv_results(args.sparse_kv_results)
        rows += r
        extra.update(x)

    if args.rows_out:
        with open(args.rows_out, "w") as f:
            json.dump({"rows": rows, "extra": extra}, f)
        return
    summarize(rows)
    if not args.no_save:
        extra.update({"model": args.model, "steps": args.steps, "warmup": args.warmup,
                      "block_size": args.block_size, "gpu": torch.cuda.get_device_name()})
        save_result(args.bench_name, rows, label=args.label or args.model.split("/")[-1],
                    extra=extra)


if __name__ == "__main__":
    main()
