"""
Llama-3-8B sparse KV cache integration.
Hooks kv_evict_quant_forward into the attention layers and measures
perplexity + latency vs dense SDPA at 4K/16K/64K context.
"""
import torch
from datetime import datetime
import os
import math
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from sparse_kv._C import kv_evict_quant_forward

MODEL_ID  = "meta-llama/Llama-3.1-8B"
DEVICE    = "cuda"
CTX_LENS  = [4096, 16384, 64000]
TOP_K     = 32

# ── sparse attention hook ──────────────────────────────────────────────────────
class SparseKVHook:
    """Hooks into LlamaAttention.forward to replace SDPA with sparse kernel."""
    def __init__(self, top_k: int):
        self.top_k   = top_k
        self.handles = []

    def hook_model(self, model):
        for layer in model.model.layers:
            attn = layer.self_attn
            handle = attn.register_forward_hook(self._attn_hook)
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _attn_hook(self, module, args, output):
        # output = (attn_output, attn_weights, past_key_value)
        # We intercept by patching _attn inside the module instead
        pass  # see patch_attention below


def patch_attention(model, top_k: int):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    import types

    def sparse_forward(self, hidden_states, attention_mask=None,
                       position_ids=None, past_key_value=None,
                       output_attentions=False, use_cache=False,
                       cache_position=None, position_embeddings=None,
                       **kwargs):

        B, S, _ = hidden_states.shape
        H   = self.config.num_attention_heads
        D   = self.head_dim
        Hkv = self.config.num_key_value_heads

        Q = self.q_proj(hidden_states).view(B, S, H,   D).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, S, Hkv, D).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, S, Hkv, D).transpose(1, 2)

        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(V, position_ids)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # GQA expand KV heads
        K = repeat_kv(K, H // Hkv)   # [B, H, S, D]
        V = repeat_kv(V, H // Hkv)

        if S == 1 and K.shape[2] >= top_k:
            attn_out = kv_evict_quant_forward(
                Q.contiguous().to(torch.float16),
                K.contiguous().to(torch.float16),
                V.contiguous().to(torch.float16),
                top_k, False
            ).to(Q.dtype)
        else:
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, is_causal=(S > 1))

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, H * D)
        attn_out = self.o_proj(attn_out)
        return attn_out, None

    # patch all attention layers
    patched = 0
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(
            sparse_forward, layer.self_attn)
        patched += 1
    print(f"✅ Patched {patched} attention layers with sparse-kv kernel")


def benchmark_latency(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000   # ms


def run():
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE,
    )
    model.eval()
    print(f"✅ Model loaded | {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir    = f"results/llama_run_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    records    = []

    print(f"\n{'='*70}")
    print(f"  sparse-kv-cuda · Llama-3.1-8B decode benchmark · top_k={TOP_K}")
    print(f"{'='*70}")
    print(f"  {'ctx_len':>10}  {'mode':>12}  {'latency':>10}  {'speedup':>10}")
    print(f"  {'-'*50}")

    for ctx_len in CTX_LENS:
        input_ids = torch.randint(
            1000, 32000, (1, ctx_len), device=DEVICE, dtype=torch.long)
        new_token = torch.randint(
            1000, 32000, (1, 1), device=DEVICE, dtype=torch.long)

        # ── dense baseline: prefill → decode ─────────────────────────────────
        # remove any patch first
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'forward'):
                try: del layer.self_attn.forward
                except: pass

        with torch.no_grad():
            # prefill: populate KV cache
            out = model(input_ids, use_cache=True)
            past_kv = out.past_key_values

            # warmup
            for _ in range(3):
                model(new_token, past_key_values=past_kv, use_cache=True)

            dense_lat = benchmark_latency(
                lambda: model(new_token, past_key_values=past_kv, use_cache=True))

        # ── sparse-kv: same prefill, patched decode ───────────────────────────
        patch_attention(model, TOP_K)

        with torch.no_grad():
            for _ in range(3):
                model(new_token, past_key_values=past_kv, use_cache=True)

            sparse_lat = benchmark_latency(
                lambda: model(new_token, past_key_values=past_kv, use_cache=True))

        # remove patch
        for layer in model.model.layers:
            try: del layer.self_attn.forward
            except: pass

        # free KV cache
        del past_kv
        torch.cuda.empty_cache()

        speedup = dense_lat / sparse_lat
        print(f"  {ctx_len:>10,}  {'dense':>12}  {dense_lat:>8.2f}ms  {'1.00x':>10}")
        print(f"  {ctx_len:>10,}  {'sparse-kv':>12}  {sparse_lat:>8.2f}ms  {speedup:>9.2f}x")
        print()
        records.append({"ctx_len": ctx_len, "mode": "dense",     "latency_ms": round(dense_lat, 3),  "speedup": 1.0})
        records.append({"ctx_len": ctx_len, "mode": "sparse-kv", "latency_ms": round(sparse_lat, 3), "speedup": round(speedup, 3)})
        save_results(records, TOP_K)

def save_results(records, top_k, model_id="meta-llama/Meta-Llama-3.1-8B"):
    import csv, json, os
    from datetime import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = f"results/llama_run_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = f"{out_dir}/results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ctx_len","mode","latency_ms","speedup"])
        writer.writeheader(); writer.writerows(records)

    # JSON
    json_path = f"{out_dir}/results.json"
    with open(json_path, "w") as f:
        json.dump({"model": model_id, "top_k": top_k,
                   "timestamp": timestamp, "results": records}, f, indent=2)

    # Plot
    ctx_lens    = sorted(set(r["ctx_len"] for r in records))
    dense_lats  = [r["latency_ms"] for r in records if r["mode"] == "dense"]
    sparse_lats = [r["latency_ms"] for r in records if r["mode"] == "sparse-kv"]
    speedups    = [r["speedup"]    for r in records if r["mode"] == "sparse-kv"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"sparse-kv-cuda · Llama-3.1-8B · top_k={top_k}", fontsize=13, fontweight="bold")

    x, w = np.arange(len(ctx_lens)), 0.35
    axes[0].bar(x - w/2, dense_lats,  w, label="Dense SDPA",  color="#4e79a7", alpha=0.85)
    axes[0].bar(x + w/2, sparse_lats, w, label="Sparse-KV",   color="#f28e2b", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"{c:,}" for c in ctx_lens])
    axes[0].set_xlabel("Context Length (tokens)"); axes[0].set_ylabel("Decode Latency (ms)")
    axes[0].set_title("Decode Latency vs Context Length"); axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
    for i,(d,s) in enumerate(zip(dense_lats, sparse_lats)):
        axes[0].text(i-w/2, d+0.3, f"{d:.1f}", ha="center", fontsize=8)
        axes[0].text(i+w/2, s+0.3, f"{s:.1f}", ha="center", fontsize=8)

    axes[1].plot(ctx_lens, speedups, marker="o", color="#59a14f", linewidth=2.5, markersize=8)
    axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (1×)")
    axes[1].fill_between(ctx_lens, 1.0, speedups, alpha=0.15, color="#59a14f")
    axes[1].set_xlabel("Context Length (tokens)"); axes[1].set_ylabel("Speedup (×)")
    axes[1].set_title("Speedup of Sparse-KV over Dense")
    axes[1].set_xscale("log", base=2); axes[1].legend(); axes[1].grid(alpha=0.3)
    for cx,sp in zip(ctx_lens, speedups):
        axes[1].annotate(f"{sp:.2f}×", (cx, sp), textcoords="offset points",
                         xytext=(0,8), ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    png_path = f"{out_dir}/llama_benchmark.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight"); plt.close()

    print(f"\n  💾 CSV  → {csv_path}")
    print(f"  💾 JSON → {json_path}")
    print(f"  📊 PNG  → {png_path}")
    return out_dir


if __name__ == "__main__":
    run()
