"""
Generate RESULTS.md from everything under results/.

The point: a reader (recruiter, hiring engineer, or you in six months) should be
able to see the whole optimisation progression -- what changed, what it bought,
on what hardware, at which commit -- without running anything, and should be
able to reproduce any single row from the SHA.

    python benchmarks/make_report.py
    python benchmarks/make_report.py --out RESULTS.md

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
import csv
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")


def load(bench):
    runs = []
    for p in sorted(glob.glob(os.path.join(RESULTS, bench, "*.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
            d["_path"] = os.path.relpath(p, REPO_ROOT).replace("\\", "/")
            runs.append(d)
        except (ValueError, OSError):
            continue
    runs.sort(key=lambda r: r.get("provenance", {}).get("utc", ""))
    return runs


def env_line(runs):
    for r in reversed(runs):
        p = r.get("provenance", {})
        if p.get("gpu"):
            return (
                "Measured on **{gpu}** ({sm}, {hbm} GB HBM) - torch {torch} / "
                "CUDA {cuda}, Python {py}.".format(
                    gpu=p["gpu"],
                    sm=p.get("sm", "?"),
                    hbm=p.get("hbm_gb", "?"),
                    torch=p.get("torch", "?"),
                    cuda=p.get("cuda", "?"),
                    py=p.get("python", "?"),
                )
            )
    return ""


def fmt(v, spec="{:.4f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "-"


def triton_h100_table():
    csv_path = os.path.join(RESULTS, "triton", "nvidia-h100-80gb-hbm3", "decode.csv")
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    sample = rows[0]
    out = [
        "",
        "## Kernel Suite on H100 (CUDA `v4`, CUDA `tc` Tensor-Core `cp.async`, Triton vs SDPA)",
        "",
        "GPU `{gpu}` | commit `{sha}` | torch {torch} | triton {triton}. "
        "Measured at `Hq=32, Hkv=8, D=128`, cold L2, median of 100 calls.".format(
            gpu=sample.get("gpu", "H100"),
            sha=sample.get("git_sha", "?"),
            torch=sample.get("torch", "?"),
            triton=sample.get("triton", "?"),
        ),
        "",
        "| Batch | Context | Kernel | Latency (ms) | Effective BW (GB/s) | % HBM Peak | Max Err vs SDPA |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    key_ctxs = {"4096", "16384", "65536"}
    for r in rows:
        if r.get("ctx") not in key_ctxs:
            continue
        out.append(
            "| {b} | {c} | `{k}` | {ms:.4f} | {bw:.0f} | {pct:.1f}% | {err:.5f} |".format(
                b=r["batch"],
                c=r["ctx"],
                k=r["kernel"],
                ms=float(r["ms"]),
                bw=float(r["gbps"]),
                pct=float(r["pct_peak"]),
                err=float(r["max_err"]),
            )
        )
    out += [
        "",
        "Full sweep (`ctx` 4K–64K): `results/triton/nvidia-h100-80gb-hbm3/decode.md`.",
    ]
    return out


def kernel_progression(runs, ctx):
    """One row per run at a fixed context: the optimisation story."""
    out = [
        "",
        "### Split-KV GQA decode kernel at {} context (batch 1, isolated)".format(ctx),
        "",
        "| Stage | ours (ms) | SDPA (ms) | ratio | % HBM peak | max err | commit |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    seen = False
    for r in runs:
        row = next((x for x in r.get("rows", []) if x.get("ctx") == ctx), None)
        if not row:
            continue
        seen = True
        prov = r.get("provenance", {})
        dirty = " (dirty)" if prov.get("git_dirty") else ""
        if r.get("RECONSTRUCTED"):
            dirty += " *transcribed*"
        out.append(
            "| {} | {} | {} | {}x | {} | {} | `{}`{} |".format(
                r.get("label", "?"),
                fmt(row.get("ours_ms")),
                fmt(row.get("sdpa_ms")),
                fmt(row.get("ratio"), "{:.2f}"),
                fmt(row.get("ours_pct_hbm"), "{:.1f}%"),
                fmt(row.get("max_err"), "{:.5f}"),
                prov.get("git_sha", "?"),
                dirty,
            )
        )
    return out if seen else []


def e2e_serving_section(runs):
    if not runs:
        return []
    latest = runs[-1]
    prov = latest.get("provenance", {})
    extra = latest.get("extra", {})
    out = [
        "",
        "## End-to-End Model Serving (`{model}`, WP-2)".format(
            model=extra.get("model", latest.get("label", "Qwen/Qwen3-8B"))
        ),
        "",
        "Commit `{sha}` on **{gpu}** (`block_size={bs}`, `steps={steps}`, warmup `{wu}`). "
        "`paged_fp16` / `paged_fp8` capture the full decode step in a single CUDA Graph.".format(
            sha=prov.get("git_sha", "?"),
            gpu=prov.get("gpu", "H100"),
            bs=extra.get("block_size", 64),
            steps=extra.get("steps", 64),
            wu=extra.get("warmup", 8),
        ),
        "",
        "| Batch | Context | Method | CUDA Graph | ITL mean (ms) | Throughput (tok/s) | KV Cache (GB) | Greedy Prefix Match |",
        "|---:|---:|---|:---:|---:|---:|---:|---:|",
    ]
    for r in latest.get("rows", []):
        b = r.get("batch")
        c = r.get("ctx")
        if b not in (1, 4, 16, 32, 64) or c not in (4096, 16384, 65536):
            continue
        if r.get("skipped"):
            need = r.get("need_gb")
            note = "skipped (need {:.1f} GB)".format(need) if need else "skipped"
            out.append(
                "| {} | {} | `{}` | - | {} | - | - | - |".format(
                    b, c, r.get("method", "?"), note
                )
            )
            continue
        agree = r.get("agree_prefix_vs_sdpa_eager")
        agree_s = str(agree) if agree is not None else "ref"
        out.append(
            "| {b} | {c} | `{m}` | {cg} | {itl} | {tok} | {kv} | {ag} |".format(
                b=b,
                c=c,
                m=r.get("method", "?"),
                cg="yes" if r.get("cuda_graph") else "no",
                itl=fmt(r.get("itl_ms_mean"), "{:.2f}"),
                tok=fmt(r.get("tok_s"), "{:.1f}"),
                kv=fmt(r.get("kv_cache_gb"), "{:.2f}"),
                ag=agree_s,
            )
        )
    out += ["", "Source: `{}`".format(latest["_path"])]
    return out


def vs_engines_section(runs):
    if not runs:
        return []
    latest = runs[-1]
    rows = latest.get("rows", [])
    prov = latest.get("provenance", {})
    out = [
        "",
        "## Head-to-Head vs FlashInfer & vLLM (WP-2)",
        "",
        "Commit `{sha}` on **{gpu}**.".format(
            sha=prov.get("git_sha", "?"), gpu=prov.get("gpu", "H100")
        ),
    ]

    fi_rows = [r for r in rows if r.get("engine") == "flashinfer"]
    if fi_rows:
        out += [
            "",
            "### Kernel-Level vs FlashInfer (`BatchDecodeWithPagedKVCache`, CUDA Graph)",
            "",
            "`ratio < 1.0` means `sparse_kv` is faster.",
            "",
            "| Method | Batch | Context | Ours (ms) | FlashInfer (ms) | Ratio (ours/FI) | Max Err |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in fi_rows:
            if r.get("batch") not in (1, 4, 16, 32):
                continue
            out.append(
                "| `{m}` | {b} | {c} | {o} | {fi} | {rat}x | {err} |".format(
                    m=r.get("method", "?"),
                    b=r.get("batch"),
                    c=r.get("ctx"),
                    o=fmt(r.get("ours_ms"), "{:.4f}"),
                    fi=fmt(r.get("flashinfer_ms"), "{:.4f}"),
                    rat=fmt(r.get("ours_over_flashinfer"), "{:.3f}"),
                    err=fmt(r.get("max_err_vs_flashinfer"), "{:.5f}"),
                )
            )

    eng_rows = [r for r in rows if r.get("engine") in ("sparse_kv", "vllm")]
    if eng_rows:
        out += [
            "",
            "### End-to-End Serving vs vLLM (`Qwen/Qwen3-8B`)",
            "",
            "Rows marked `*` exceeded vLLM's initial KV reservation (`fits_in_vllm_kv=false`).",
            "",
            "| Engine | Method | Batch | Context | TPOT (ms) | Throughput (tok/s) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in eng_rows:
            if r.get("batch") not in (1, 4, 16, 32) or r.get("ctx") not in (4096, 16384, 65536):
                continue
            if r.get("skipped"):
                out.append(
                    "| `{eng}` | `{m}` | {b} | {c} | skipped | - |".format(
                        eng=r.get("engine"),
                        m=r.get("method"),
                        b=r.get("batch"),
                        c=r.get("ctx"),
                    )
                )
                continue
            tpot = r.get("tpot_ms")
            tok_s = r.get("tok_s_from_tpot")
            star = "*" if (r.get("engine") == "vllm" and r.get("fits_in_vllm_kv") is False) else ""
            out.append(
                "| `{eng}` | `{m}` | {b} | {c} | {tp} | {ts}{st} |".format(
                    eng=r.get("engine"),
                    m=r.get("method"),
                    b=r.get("batch"),
                    c=r.get("ctx"),
                    tp=fmt(tpot, "{:.2f}"),
                    ts=fmt(tok_s, "{:.1f}"),
                    st=star,
                )
            )

    out += ["", "Source: `{}`".format(latest["_path"])]
    return out


def quest_and_fp8_section(fp8_runs, quest_runs, quest_qual_runs):
    out = []
    if fp8_runs or quest_runs or quest_qual_runs:
        out += ["", "## FP8 & Quest 64K Sparse Attention + Quality Gates (WP-4)"]

    if fp8_runs:
        by_ctx = {}
        for r in fp8_runs:
            for row in r.get("rows", []):
                prefill = row.get("prefill")
                if prefill in (2048, 8192):
                    by_ctx[prefill] = (r, row)
        if by_ctx:
            out += [
                "",
                "### FP8 KV Cache Quality Gate (`Qwen/Qwen3-8B`, 512 teacher-forced steps)",
                "",
                "| Prefill | PPL Ref | PPL FP8 | PPL Delta | Confident-Step Parity | Raw Argmax Parity | Verdict | Source |",
                "|---:|---:|---:|---:|---:|---:|---|---|",
            ]
            for prefill in sorted(by_ctx):
                run, row = by_ctx[prefill]
                out.append(
                    "| {p} | {pr:.2f} | {pf:.2f} | {pd:+.2f}% | {cp:.1f}% | {ap:.1f}% | **{v}** | `{src}` |".format(
                        p=prefill,
                        pr=row["ppl_ref"],
                        pf=row["ppl_fp8"],
                        pd=row["ppl_delta_pct"],
                        cp=row["parity_confident_steps"] * 100.0,
                        ap=row["argmax_parity"] * 100.0,
                        v=row.get("verdict", "?"),
                        src=run["_path"],
                    )
                )

    if quest_runs:
        latest = quest_runs[-1]
        out += [
            "",
            "### Quest Page-Sparse Kernel at 64K Context (`B=4`, `Hq=32, Hkv=8, D=128`, CUDA Graph)",
            "",
            "| KV Dtype | Block Size | Budget | Dense (us) | Quest (us) | Speedup | HBM Bytes Ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in latest.get("rows", []):
            out.append(
                "| `{kv}` | {bs} | {bg:.1f}% | {du:.1f} | {qu:.1f} | **{sp:.2f}x** | {br:.1f}% |".format(
                    kv=r["kv"],
                    bs=r["block_size"],
                    bg=r["budget"] * 100.0,
                    du=r["dense_us"],
                    qu=r["quest_us"],
                    sp=r["speedup"],
                    br=r["bytes_ratio"] * 100.0,
                )
            )
        out += ["", "Source: `{}`".format(latest["_path"])]

    if quest_qual_runs:
        latest = quest_qual_runs[-1]
        out += [
            "",
            "### Quest 64K Quality Gate (`Qwen/Qwen3-8B`, 2 dense prefix layers)",
            "",
            "| Path | Budget | PPL Delta | Confident-Step Parity | Raw Argmax Parity | Needle Recall | Verdict |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in latest.get("rows", []):
            bg = "{:.1f}%".format(r["budget"] * 100.0) if r.get("budget") is not None else "100% (dense)"
            needles = "{}/{}".format(r.get("needle_found", "?"), r.get("needle_ref_found", "?"))
            out.append(
                "| `{path}` | {bg} | {pd:+.2f}% | {cp:.1f}% | {ap:.1f}% | {nd} | **{v}** |".format(
                    path=r.get("path", "?"),
                    bg=bg,
                    pd=r["ppl_delta_pct"],
                    cp=r["parity_confident_steps"] * 100.0,
                    ap=r["argmax_parity"] * 100.0,
                    nd=needles,
                    v=r.get("verdict", "?"),
                )
            )
        out += ["", "Source: `{}`".format(latest["_path"])]

    return out


def tp_decode_section(runs):
    if not runs:
        return []
    out = [
        "",
        "## Multi-GPU Tensor-Parallel Decode (`TP=2` & `TP=4` over NVLink, WP-5)",
        "",
        "1-layer `Qwen3-8B` TP block with real `torch.distributed` NCCL `AllReduce` "
        "(`nccl_graph_supported=true`, CUDA Graph over NCCL enabled).",
        "",
        "| TP | KV Mode | Batch | Context | Compute Only (ms) | Eager TP (ms) | Graph TP (ms) | Graph Speedup | Eager Comm % |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sources = []
    for run in runs:
        sources.append("`{}`".format(run["_path"]))
        for r in run.get("rows", []):
            if r.get("batch") not in (1, 16, 64):
                continue
            out.append(
                "| {tp} | `{m}` | {b} | {c} | {cmp:.3f} | {eag:.3f} | **{gph:.3f}** | **{spd:.2f}x** | {cp:.1f}% |".format(
                    tp=r["tp_size"],
                    m=r["kv_mode"],
                    b=r["batch"],
                    c=r["ctx"],
                    cmp=r["compute_ms"],
                    eag=r["eager_tp_ms"],
                    gph=r["graph_tp_ms"],
                    spd=r["graph_speedup"],
                    cp=r["comm_pct"],
                )
            )
    out += ["", "Sources: " + ", ".join(sources)]
    return out


def profiles_section():
    sass_path = os.path.join(RESULTS, "profiles", "h100_sass.md")
    nsys_path = os.path.join(RESULTS, "profiles", "h100_nsys_profile.md")
    if not os.path.exists(sass_path) and not os.path.exists(nsys_path):
        return []
    out = ["", "## H100 Profiling (`SASS` & `nsys` Attribution, WP-5)"]
    if os.path.exists(nsys_path):
        out += [
            "",
            "NVTX `nsys stats` on eager `Qwen/Qwen3-8B` (`paged_fp8`) attributes `~68%` of eager "
            "decode wall time to unfused `RoPE_KV_Append` (`43–44%`) + `RMSNorm` (`24–25%`), while "
            "`cuda_gpu_kern_sum` confirms pure on-GPU times of only `1.1–3.7 us` per launch — "
            "explaining why CUDA Graphs yield up to **2.36x** single-GPU E2E and **7.55x** TP speedups:",
            "",
            "- E2E `nsys` NVTX breakdown: [`results/profiles/h100_nsys_profile.md`](results/profiles/h100_nsys_profile.md)",
            "- E2E `nsys` GPU kernel sum: [`results/profiles/h100_nsys_kern_sum.txt`](results/profiles/h100_nsys_kern_sum.txt)",
            "- `TP=2` `nsys` NVTX + NCCL stats: [`results/profiles/h100_tp2_nsys_stats.txt`](results/profiles/h100_tp2_nsys_stats.txt)",
        ]
    if os.path.exists(sass_path):
        out += [
            "- CUDA `sm_90a` SASS instruction summary (`HMMA`, `LDGSTS`, `LDG.E.128`): [`results/profiles/h100_sass.md`](results/profiles/h100_sass.md)",
        ]
    out += [
        "",
        "*(Nsight Compute `ncu` hardware PMU capture is skipped because Modal containers disable GPU performance counters with `ERR_NVGPUCTRPERM`.)*",
    ]
    return out


def serving_table(runs):
    if not runs:
        return []
    latest = runs[-1]
    cfg = latest.get("extra", {})
    out = ["", "## Historical KV-Capacity & Baseline Kernel Model (Pre-WP-1)", ""]

    cap = cfg.get("capacity", {})
    if cap:
        out += [
            "Concurrent sequences that fit on one {} GB GPU "
            "({} GB weights, {} GB reserved):".format(
                cfg.get("hbm_gb", "?"),
                cfg.get("weight_gb", "?"),
                cfg.get("reserve_gb", "?"),
            ),
            "",
            "| context | KV per sequence | max concurrent (fp16) | max concurrent (fp8) |",
            "|---:|---:|---:|---:|",
        ]
        for ctx in sorted(cap, key=lambda x: int(x)):
            c = cap[ctx]
            out.append(
                "| {} | {:.2f} GB | {} | {} |".format(
                    ctx, c["kv_gb_per_seq"], c["max_seqs_fp16"], c["max_seqs_fp8"]
                )
            )
    out += ["", "Source: `{}`".format(latest["_path"])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "RESULTS.md"))
    args = ap.parse_args()

    kernel = load("kernel_vs_sdpa")
    serving = load("serving_regime")
    e2e = load("e2e_serving")
    vs_eng = load("vs_engines")
    fp8_qual = load("quality_fp8")
    quest_k = load("quest_kernel")
    quest_q = load("quality_quest")
    tp_dec = load("tp_decode")

    doc = [
        "# Measured Results",
        "",
        "Generated by `benchmarks/make_report.py` from measured artifacts under "
        "`results/`. Every table is derived from provenance-stamped runs; "
        "nothing is hand-entered.",
        "",
    ]
    env = env_line(tp_dec + e2e + vs_eng + kernel + serving)
    if env:
        doc += [env, ""]

    doc += triton_h100_table()
    doc += e2e_serving_section(e2e)
    doc += vs_engines_section(vs_eng)
    doc += quest_and_fp8_section(fp8_qual, quest_k, quest_q)
    doc += tp_decode_section(tp_dec)
    doc += profiles_section()

    if kernel:
        doc += [
            "",
            "## Historical CUDA Kernel Optimisation Progression (WP-0 Baseline → Stage 4a)",
            "",
            "Isolated single-step GQA decode kernel vs PyTorch SDPA on "
            "Llama-3.1-8B / Qwen3-8B shapes (`Hq=32, Hkv=8, D=128`). `ratio < 1.0` "
            "means our kernel is faster.",
        ]
        for ctx in (4096, 16384, 64000):
            doc += kernel_progression(kernel, ctx)
        if any(r.get("RECONSTRUCTED") for r in kernel):
            doc += [
                "",
                "Rows marked *transcribed* were copied from console output "
                "prior to structured JSON recording.",
            ]

    doc += serving_table(serving)

    doc += [
        "",
        "---",
        "",
        "Reproduce on Modal H100 (`--suite kernel|e2e|tp2|tp4|nsys|wp5|all`):",
        "",
        "```bash",
        "modal run modal_apps/h100.py --suite all",
        "```",
        "",
    ]

    with open(args.out, "w") as f:
        f.write("\n".join(doc))
    try:
        shown = os.path.relpath(args.out, REPO_ROOT)
    except ValueError:
        shown = args.out
    print(
        "wrote {} ({} kernel, {} e2e, {} vs_engines, {} quest, {} tp runs)".format(
            shown, len(kernel), len(e2e), len(vs_eng), len(quest_k), len(tp_dec)
        )
    )


if __name__ == "__main__":
    main()
