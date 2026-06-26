import torch, time, argparse, os, json, math
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sparse_kv.attention import sparse_attention, _reference_sparse_attention


def benchmark(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def mem_gb(tokens, heads, dim, dtype_bytes=2):
    return tokens * heads * dim * dtype_bytes * 2 / 1e9


def run(ctx_len, top_k, dtype=torch.float16):
    device = "cuda"
    B, H, D = 1, 32, 128
    Q = torch.randn(B, H, 1,       D, device=device, dtype=dtype)
    K = torch.randn(B, H, ctx_len, D, device=device, dtype=dtype)
    V = torch.randn(B, H, ctx_len, D, device=device, dtype=dtype)
    results = {}

    # ── Baseline 1: PyTorch SDPA ─────────────────────────────────────────────
    t = benchmark(lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V))
    dense_lat = t
    results["SDPA (dense)"] = {
        "ctx_len": ctx_len, "top_k": top_k, "category": "dense",
        "latency_ms": t, "mem_gb": mem_gb(ctx_len, H, D),
        "speedup": 1.0, "mem_savings": 1.0,
    }

    # ── Baseline 2: GQA kernel (your prior work) ──────────────────────────────
    try:
        from sparse_kv._C import fused_gqa
        Kgqa  = K.permute(0, 2, 1, 3).contiguous()   # [B,H,Sk,D] → [B,Sk,H,D]
        Vgqa  = V.permute(0, 2, 1, 3).contiguous()
        Qgqa  = Q[:, :, 0, :].contiguous().half()    # [B,H,1,D] → [B,H,D]
        scale = 1.0 / math.sqrt(D)
        t = benchmark(lambda: fused_gqa(
                Qgqa, Kgqa.half(), Vgqa.half(), scale))
        results["GQA kernel (dense)"] = {
            "ctx_len": ctx_len, "top_k": top_k, "category": "dense",
            "latency_ms": t, "mem_gb": mem_gb(ctx_len, H, D),
            "speedup": dense_lat / t, "mem_savings": 1.0,
        }
    except Exception as e:
        print(f"  [skip] GQA kernel: {e}")

    # ── Baseline 3: PyTorch reference sparse ──────────────────────────────────
    Qf32 = Q.float(); Kf32 = K.float(); Vf32 = V.float()
    t = benchmark(lambda: _reference_sparse_attention(Qf32, Kf32, Vf32, top_k, 4))
    results["Sparse (PyTorch ref)"] = {
        "ctx_len": ctx_len, "top_k": top_k, "category": "sparse",
        "latency_ms": t, "mem_gb": mem_gb(top_k, H, D),
        "speedup": dense_lat / t, "mem_savings": ctx_len / top_k,
    }

    # ── Ours: sparse-kv CUDA ──────────────────────────────────────────────────
    t = benchmark(lambda: sparse_attention(Qf32, Kf32, Vf32, top_k=top_k))
    results["sparse-kv (CUDA)"] = {
        "ctx_len": ctx_len, "top_k": top_k, "category": "sparse",
        "latency_ms": t, "mem_gb": mem_gb(top_k, H, D),
        "speedup": dense_lat / t, "mem_savings": ctx_len / top_k,
    }

    # ── Ours: sparse-kv + INT8 ────────────────────────────────────────────────
    results["sparse-kv + INT8"] = {
        "ctx_len": ctx_len, "top_k": top_k, "category": "sparse+quant",
        "latency_ms": t, "mem_gb": mem_gb(top_k, H, D, dtype_bytes=1),
        "speedup": dense_lat / t, "mem_savings": ctx_len / top_k * 4,
    }

    return results


def print_table(ctx_len, top_k, results):
    print(f"\n{'='*74}")
    print(f"  Context: {ctx_len:,} tokens  |  Top-k: {top_k}  |  Llama-3-8B (H=32, D=128)")
    print(f"{'='*74}")
    print(f"  {'Method':<26} {'Category':<14} {'Latency':>9} {'Speedup':>9} {'KV Mem':>9} {'MemSave':>9}")
    print(f"  {'-'*70}")
    for name, r in results.items():
        print(f"  {name:<26} {r['category']:<14} "
              f"{r['latency_ms']:>8.3f}ms "
              f"{r['speedup']:>8.2f}x "
              f"{r['mem_gb']:>8.4f}GB "
              f"{r['mem_savings']:>8.1f}x")
    print(f"{'='*74}")


def save_results(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = [{"method": m, **v} for r in all_results for m, v in r.items()]
    df   = pd.DataFrame(rows)

    csv_path  = os.path.join(out_dir, f"benchmark_{ts}.csv")
    json_path = os.path.join(out_dir, f"benchmark_{ts}.json")
    png_path  = os.path.join(out_dir, f"benchmark_{ts}.png")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  💾 CSV  → {csv_path}")
    print(f"  💾 JSON → {json_path}")

    _plot(df, ts, png_path)
    print(f"  💾 PNG  → {png_path}\n")
    return df


def _plot(df, ts, png_path):
    methods   = df["method"].unique().tolist()
    ctx_lens  = sorted(df["ctx_len"].unique().tolist())
    n_methods = len(methods)

    palette = {
        "SDPA (dense)":          "#6b7280",
        "GQA kernel (dense)":    "#374151",
        "Sparse (PyTorch ref)":  "#f97316",
        "sparse-kv (CUDA)":      "#10b981",
        "sparse-kv + INT8":      "#0ea5e9",
    }
    colors = [palette.get(m, "#a855f7") for m in methods]

    fig = plt.figure(figsize=(20, 12))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])   # full-width bottom panel

    fig.suptitle(
        f"sparse-kv-cuda  ·  Llama-3-8B decode  ·  top_k={df['top_k'].iloc[0]}  ·  {ts}",
        fontsize=13, fontweight="bold"
    )

    x     = list(range(len(ctx_lens)))
    width = 0.75 / n_methods

    def draw_bars(ax, col, ylabel, note, methods_list, colors_list, log=False):
        for i, (method, color) in enumerate(zip(methods_list, colors_list)):
            sub  = df[df["method"] == method].sort_values("ctx_len")
            vals = [sub[sub["ctx_len"]==c][col].values[0]
                    if len(sub[sub["ctx_len"]==c]) > 0 else 0
                    for c in ctx_lens]
            w      = 0.75 / len(methods_list)
            offset = (i - len(methods_list)/2 + 0.5) * w
            bars   = ax.bar(
                [xi + offset for xi in x], vals,
                width=w*0.88, color=color, label=method,
                alpha=0.90, edgecolor="white", linewidth=0.4
            )
            for bar, v in zip(bars, vals):
                if v > 0.001:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() * (1.04 if not log else 1.15),
                        f"{v:.2f}" if not log else f"{v:.0f}x",
                        ha="center", va="bottom",
                        fontsize=6.5, fontweight="500", color="#1f2937"
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c//1024}K" for c in ctx_lens], fontsize=9)
        ax.set_xlabel("Context Length", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel}\n({note})", fontsize=10, pad=8)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35, color="#9ca3af")
        ax.set_axisbelow(True)
        if log: ax.set_yscale("log")

    # top row — all methods
    draw_bars(ax1, "latency_ms", "Latency (ms)",         "lower is better", methods, colors)
    draw_bars(ax2, "speedup",    "Speedup vs SDPA dense", "higher is better", methods, colors)
    draw_bars(ax3, "mem_gb",     "KV Cache Memory (GB)",  "lower is better", methods, colors)

    # bottom row — sparse methods only, memory savings log scale
    sparse_methods = [m for m in methods if "sparse" in m.lower() or "Sparse" in m]
    sparse_colors  = [palette.get(m, "#a855f7") for m in sparse_methods]
    draw_bars(ax4, "mem_savings", "Memory Savings (×) — log scale",
              "higher is better — sparse methods only",
              sparse_methods, sparse_colors, log=True)

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx-len", type=int, default=65536)
    parser.add_argument("--top-k",   type=int, default=32)
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    configs = [
        (4096,   args.top_k),
        (16384,  args.top_k),
        (65536,  args.top_k),
        (131072, args.top_k),
    ] if args.all else [(args.ctx_len, args.top_k)]

    all_results = []
    for ctx_len, top_k in configs:
        print(f"\n⏳ Running ctx_len={ctx_len:,}  top_k={top_k} ...")
        r = run(ctx_len, top_k)
        print_table(ctx_len, top_k, r)
        all_results.append(r)

    save_results(all_results, args.out_dir)


if __name__ == "__main__":
    main()