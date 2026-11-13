"""Real end-to-end decode benchmark (WP-2): HF weights, paged KV pool, CUDA graphs, measured TPOT.

Replaces the analytical step_ms* / tok_s* projections of bench_serving_regime.py with a real
model: every layer's projections, norms, RoPE, MLP, lm_head and greedy sampling run on the GPU,
one decode step for the whole batch is captured as a single CUDA graph, and the graph is replayed
back to back. Inter-token latency (ITL = TPOT for a static batch) is timed per step with CUDA
events; throughput is B * steps / wall time.

Methods (same weights, same prompt, same prefill code; only the KV cache and attention differ):
  sdpa_eager    contiguous static cache, F.scaled_dot_product_attention over the live prefix,
                plain eager Python loop (no graph)
  sdpa_compile  same cache, masked SDPA over the whole buffer, torch.compile'd step + CUDA graph
  paged_fp16    sparse_kv paged pool (fp16), torch.ops.sparse_kv append + decode, CUDA graph
  paged_fp8     sparse_kv paged pool (e4m3 + per-page scales), CUDA graph

Prefill: once per ctx, one prompt of `ctx` random tokens runs through the real model (layer by
layer, causal flash/mem-efficient SDPA); its per-layer K/V are kept in host memory and written
into each (B, method)'s cache, then replicated to all B sequences (B full copies: decode really
reads B * ctx tokens of KV per layer). prefill_ms_1seq is that single-sequence prefill (GPU time
up to the last-token logits, before the K/V move to host), not a batched TTFT.

CUDA graph rules followed here (see sparse_kv/ops.py):
  * paged_gqa_decode gets max_seq_len = ctx + decode steps (the graph's capacity) explicitly, so
    nothing syncs device -> host and the split grid is fixed for the graph;
  * the whole step is warmed up (Triton JIT + autotune for this exact (B, max_seq_len) bucket)
    with every sequence inactive (appends skipped, positions frozen) before capture;
  * capture_cuda_graph() fails loudly if any Triton autotuner fired during capture.

Run:
  python benchmarks/bench_e2e_serving.py                               # Qwen/Qwen3-8B, full sweep
  python benchmarks/bench_e2e_serving.py --model meta-llama/Llama-3.1-8B --batches 1,16
  python benchmarks/bench_e2e_serving.py --model tiny-random --batches 1,4 --contexts 256,1024 \
      --steps 8 --warmup 2 --no-save                                   # smoke test, no download

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
import gc
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import save_result  # noqa: E402
from sparse_kv import ops as kv_ops  # noqa: E402
from triton_kernels.gqa_decode import KV_PAGE_TOKENS  # noqa: E402
from triton_kernels.paged_kv import allocate_paged_kv_cache, apply_rope  # noqa: E402

METHODS = ("sdpa_eager", "sdpa_compile", "paged_fp16", "paged_fp8")
GRAPHED = ("sdpa_compile", "paged_fp16", "paged_fp8")


# --------------------------------------------------------------------------------- model load

def long_context_overrides(cfg, needed_len):
    """Config changes needed to run `needed_len` positions. Qwen3 ships 32K native (40960 in
    config) and documents YaRN x4 for longer contexts; models already long enough get none."""
    native = getattr(cfg, "max_position_embeddings", None)
    if native is None or needed_len <= native or getattr(cfg, "rope_scaling", None):
        return {}
    if not getattr(cfg, "model_type", "").startswith("qwen"):
        print("warning: {} positions > max_position_embeddings {}; no RoPE override known for "
              "model_type {}".format(needed_len, native, cfg.model_type))
        return {}
    orig = 32768 if cfg.model_type == "qwen3" else native
    factor = float(max(2, 2 ** math.ceil(math.log2(needed_len / orig))))
    return {"rope_scaling": {"rope_type": "yarn", "factor": factor,
                             "original_max_position_embeddings": orig},
            "max_position_embeddings": int(orig * factor)}


def load_model(name, max_len, device="cuda"):
    """Returns (hf_model, rope_overrides). 'tiny-random' builds a 4-layer Llama with random
    weights (no download) for smoke tests."""
    from transformers import AutoConfig, AutoModelForCausalLM
    if name == "tiny-random":
        from transformers import LlamaConfig
        cfg = LlamaConfig(vocab_size=32000, hidden_size=1024, intermediate_size=2816,
                          num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                          max_position_embeddings=max(4096, max_len))
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
        return model.to(device).eval().requires_grad_(False), {}
    cfg = AutoConfig.from_pretrained(name)
    overrides = long_context_overrides(cfg, max_len)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    # The decode kernels take fp16 q/K/V, so the whole model runs in fp16.
    model = AutoModelForCausalLM.from_pretrained(name, config=cfg, torch_dtype=torch.float16)
    return model.to(device).eval().requires_grad_(False), overrides


def rms_norm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


class Model:
    """Our forward over the HF module's weights (Llama / Qwen2 / Qwen3 layout)."""

    def __init__(self, hf, max_pos):
        m, cfg = hf.model, hf.config
        self.hq, self.hkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.d = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.hq
        self.hidden = cfg.hidden_size
        self.vocab = cfg.vocab_size
        self.eps = cfg.rms_norm_eps
        self.scale = self.d ** -0.5
        self.n_layers = len(m.layers)
        self.embed_w = m.embed_tokens.weight
        self.norm_w = m.norm.weight
        self.lm_head = hf.lm_head.weight

        def norm_w(mod, name):
            sub = getattr(mod, name, None)
            return None if sub is None else sub.weight

        self.layers = []
        for layer in m.layers:
            a, mlp = layer.self_attn, layer.mlp
            self.layers.append(dict(
                ln1=layer.input_layernorm.weight, ln2=layer.post_attention_layernorm.weight,
                q=a.q_proj.weight, qb=a.q_proj.bias, k=a.k_proj.weight, kb=a.k_proj.bias,
                v=a.v_proj.weight, vb=a.v_proj.bias, o=a.o_proj.weight, ob=a.o_proj.bias,
                qn=norm_w(a, "q_norm"), kn=norm_w(a, "k_norm"),
                gate=mlp.gate_proj.weight, up=mlp.up_proj.weight, down=mlp.down_proj.weight))

        # cos/sin from the model's own rotary module (handles llama3 / yarn scaling).
        rot = getattr(m, "rotary_emb", None) or m.layers[0].self_attn.rotary_emb
        dev = self.embed_w.device
        with torch.no_grad():
            cos, sin = rot(torch.zeros(1, device=dev, dtype=torch.float32),
                           torch.arange(max_pos, device=dev)[None])
        half = self.d // 2
        self.cos = cos[0, :, :half].float().contiguous()
        self.sin = sin[0, :, :half].float().contiguous()

    def qkv(self, l, h, positions):
        """h [T, hidden] (pre-norm residual) -> q [T, Hq, D], k/v [T, Hkv, D], RoPE applied."""
        L, T = self.layers[l], h.shape[0]
        x = rms_norm(h, L["ln1"], self.eps)
        q = F.linear(x, L["q"], L["qb"]).view(T, self.hq, self.d)
        k = F.linear(x, L["k"], L["kb"]).view(T, self.hkv, self.d)
        v = F.linear(x, L["v"], L["vb"]).view(T, self.hkv, self.d)
        if L["qn"] is not None:  # Qwen3 per-head q/k RMSNorm
            q = rms_norm(q, L["qn"], self.eps)
            k = rms_norm(k, L["kn"], self.eps)
        return (apply_rope(q, self.cos, self.sin, positions),
                apply_rope(k, self.cos, self.sin, positions), v.contiguous())

    def post_attn(self, l, h, o):
        L = self.layers[l]
        h = h + F.linear(o.reshape(h.shape[0], -1), L["o"], L["ob"])
        x = rms_norm(h, L["ln2"], self.eps)
        return h + F.linear(F.silu(F.linear(x, L["gate"])) * F.linear(x, L["up"]), L["down"])

    def logits(self, h):
        return F.linear(rms_norm(h, self.norm_w, self.eps), self.lm_head)


def causal_attention(q, k, v, scale):
    """Prefill attention for one sequence: q [T, Hq, D], k/v [T, Hkv, D] -> [T, Hq, D]."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    rep = q.shape[1] // k.shape[1]
    qh = q.transpose(0, 1)[None]
    kh = k.repeat_interleave(rep, dim=1).transpose(0, 1)[None]
    vh = v.repeat_interleave(rep, dim=1).transpose(0, 1)[None]
    # Never the math backend: it materialises T x T scores (hundreds of GB at 64K).
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=scale)
    return o[0].transpose(0, 1)


@torch.no_grad()
def prefill(M, ids, sink, chunk=4096):
    """Run one prompt through the model layer by layer; sink(l, k, v) stores each layer's K/V.
    Token-chunked projections / MLP bound the activation memory. Returns last-token logits."""
    T = ids.numel()
    pos = torch.arange(T, device=ids.device)
    h = F.embedding(ids, M.embed_w)
    for l in range(M.n_layers):
        q = h.new_empty(T, M.hq, M.d)
        k = h.new_empty(T, M.hkv, M.d)
        v = h.new_empty(T, M.hkv, M.d)
        for s in range(0, T, chunk):
            q[s:s + chunk], k[s:s + chunk], v[s:s + chunk] = M.qkv(l, h[s:s + chunk],
                                                                  pos[s:s + chunk])
        sink(l, k, v)
        o = causal_attention(q, k, v, M.scale)
        del q, k, v
        for s in range(0, T, chunk):
            h[s:s + chunk] = M.post_attn(l, h[s:s + chunk], o[s:s + chunk])
        del o
    return M.logits(h[-1:])


def _to_host(t):
    try:
        out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    except RuntimeError:  # pinning can fail under tight locked-memory limits
        out = torch.empty(t.shape, dtype=t.dtype)
    out.copy_(t)
    return out


class PrefillCache:
    """One prefill of `prompt`, shared by every (B, method) at this ctx.

    The per-layer K/V stay on the GPU while prefill is timed (no pool is allocated yet), then
    move to host memory so they do not shrink the KV pools being benchmarked."""

    def __init__(self, M, prompt):
        on_gpu = []  # (k, v) per layer

        def sink(l, k, v):
            on_gpu.append((k.clone(), v.clone()))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.logits = prefill(M, prompt, sink)
        torch.cuda.synchronize()
        self.ms = (time.perf_counter() - t0) * 1e3
        self.k, self.v = [], []
        while on_gpu:  # move layer by layer, freeing each GPU copy as it goes
            k, v = on_gpu.pop(0)
            self.k.append(_to_host(k))
            self.v.append(_to_host(v))
        torch.cuda.empty_cache()

    def fill(self, st, device):
        for l in range(len(self.k)):
            st.sink(l, self.k[l].to(device, non_blocking=True),
                    self.v[l].to(device, non_blocking=True))


# ------------------------------------------------------------------------------- decode state

class DecodeState:
    """KV cache + static buffers for one (method, B, ctx). step() is graph-capturable: it reads
    and writes only these buffers, and does its own bookkeeping (positions, slots, sampling)."""

    def __init__(self, M, method, B, ctx, gen, block_size, device, seed=0):
        self.M, self.method, self.B, self.ctx, self.gen = M, method, B, ctx, gen
        self.paged = method.startswith("paged")
        self.capacity = ctx + gen  # max tokens per sequence; = max_seq_len for the graph
        self.bs = block_size
        i64 = dict(dtype=torch.long, device=device)
        self.input_ids = torch.zeros(B, **i64)
        self.positions = torch.zeros(B, **i64)       # position of the token being processed
        self.active = torch.ones(B, dtype=torch.bool, device=device)
        self.step_idx = torch.zeros(1, **i64)
        self.token_log = torch.zeros(gen + 1, B, **i64)
        self.host_len = ctx  # sdpa_eager only: live prefix length, tracked on the host
        L = M.n_layers
        if self.paged:
            bps = -(-self.capacity // block_size)
            nb = B * bps
            self.pool = [allocate_paged_kv_cache(nb, block_size, M.hkv, M.d,
                                                 method == "paged_fp8", device)
                         for _ in range(L)]
            # Shuffled physical blocks: a fragmented pool, as after real admission/eviction.
            g = torch.Generator().manual_seed(seed)
            self.block_table = torch.randperm(nb, generator=g).view(B, bps).int().to(device)
            self.seq_lens = torch.zeros(B, dtype=torch.int32, device=device)
        else:
            shape = (B, M.hkv, self.capacity, M.d)
            self.kc = [torch.zeros(shape, dtype=torch.float16, device=device) for _ in range(L)]
            self.vc = [torch.zeros(shape, dtype=torch.float16, device=device) for _ in range(L)]
            self.b_idx = torch.arange(B, device=device)
            self.cap_range = torch.arange(self.capacity, device=device)

    def kv_bytes(self):
        if self.paged:
            return sum(t.numel() * t.element_size() for p in self.pool for t in p if t is not None)
        return sum(t.numel() * t.element_size() for t in self.kc + self.vc)

    # -- prefill: write sequence 0, then replicate to the other B - 1 sequences
    def sink(self, l, k, v):
        T = k.shape[0]
        if self.paged:
            pos = torch.arange(T, device=k.device)
            slots = self.block_table[0, pos // self.bs].long() * self.bs + pos % self.bs
            kc, vc, ks, vs = self.pool[l]
            kv_ops.reshape_and_cache(k, v, kc, vc, slots, ks, vs)
        else:
            self.kc[l][0, :, :T] = k.transpose(0, 1)
            self.vc[l][0, :, :T] = v.transpose(0, 1)

    def replicate_prefix(self):
        T = self.ctx
        if self.paged:
            n = -(-T // self.bs)
            src = self.block_table[0, :n].long()
            for p in self.pool:
                for t in p:
                    if t is None:
                        continue
                    blocks = t[src]
                    for b in range(1, self.B):
                        t.index_copy_(0, self.block_table[b, :n].long(), blocks)
        else:
            for t in self.kc + self.vc:
                t[1:, :, :T] = t[0:1, :, :T]

    def start(self, first_logits):
        tok = first_logits.argmax(-1).view(1)
        self.input_ids.copy_(tok.expand(self.B))
        self.positions.fill_(self.ctx)
        self.token_log[0].copy_(tok.expand(self.B))
        self.step_idx.fill_(1)

    # -- one decode step for the whole batch
    def step(self):
        M, B, pos = self.M, self.B, self.positions
        if self.paged:
            # Inactive (warm-up) rows append nothing, so they must not attend to slot `pos`.
            self.seq_lens.copy_(torch.where(self.active, pos + 1, pos).int())
            blk = self.block_table.gather(1, (pos // self.bs).view(B, 1)).view(B).long()
            slots = torch.where(self.active, blk * self.bs + pos % self.bs,
                                torch.full_like(pos, -1))  # inactive: append skipped
        elif self.method == "sdpa_compile":
            mask = (self.cap_range[None, :] <= pos[:, None]).view(B, 1, 1, self.capacity)
        h = F.embedding(self.input_ids, M.embed_w)
        for l in range(M.n_layers):
            q, k, v = M.qkv(l, h, pos)
            if self.paged:
                kc, vc, ks, vs = self.pool[l]
                kv_ops.reshape_and_cache(k, v, kc, vc, slots, ks, vs)
                o = kv_ops.paged_gqa_decode(q, kc, vc, self.block_table, self.seq_lens,
                                            self.capacity, scale=M.scale, k_scale=ks,
                                            v_scale=vs)
            else:
                kc, vc = self.kc[l], self.vc[l]
                kc[self.b_idx, :, pos] = k
                vc[self.b_idx, :, pos] = v
                if self.method == "sdpa_eager":
                    n = self.host_len
                    o = F.scaled_dot_product_attention(q[:, :, None], kc[:, :, :n], vc[:, :, :n],
                                                       scale=M.scale, enable_gqa=True)
                else:
                    o = F.scaled_dot_product_attention(q[:, :, None], kc, vc, attn_mask=mask,
                                                       scale=M.scale, enable_gqa=True)
                o = o.squeeze(2)
            h = M.post_attn(l, h, o)
        nxt = M.logits(h).argmax(-1)
        self.input_ids.copy_(torch.where(self.active, nxt, self.input_ids))
        self.token_log.index_copy_(0, self.step_idx.clamp(max=self.gen), nxt.view(1, B))
        adv = self.active.long()
        self.step_idx.add_(adv.max())
        self.positions.add_(adv)


# ------------------------------------------------------------------------------------ runner

def estimate_bytes(M, method, B, ctx, gen, block_size):
    cap = ctx + gen
    L, hkv, d = M.n_layers, M.hkv, M.d
    if method.startswith("paged"):
        tokens = B * -(-cap // block_size) * block_size
        elem = 1 if method == "paged_fp8" else 2
        kv = 2 * L * tokens * hkv * d * elem
        if elem == 1:
            kv += 2 * L * (tokens // KV_PAGE_TOKENS) * hkv * 4
        transient = 0
    else:
        kv = 2 * L * B * cap * hkv * d * 2
        transient = 2 * B * cap * M.hq * d * 2  # SDPA may expand GQA K/V for one layer
    # Prefill ran once per ctx, before any pool; here only one layer's cached K/V is uploaded at
    # a time (plus, for paged, the per-block gather in replicate_prefix).
    upload = 2 * 2 * ctx * hkv * d * 2
    return kv + max(upload, transient) + (1 << 30)


def run_one(M, method, B, ctx, steps, warmup, block_size, pre, device="cuda"):
    gen = warmup + steps
    st = DecodeState(M, method, B, ctx, gen, block_size, device)
    pre.fill(st, device)
    st.replicate_prefix()
    st.start(pre.logits)

    if method == "sdpa_eager":
        fn = st.step
    else:
        step = torch.compile(st.step, dynamic=False) if method == "sdpa_compile" else st.step
        # Warm up (JIT + autotune at this exact (B, max_seq_len)) with every sequence inactive:
        # appends are skipped and positions / step_idx stay frozen, so the state is untouched.
        st.active.fill_(False)
        graph, _ = kv_ops.capture_cuda_graph(step, warmup_iters=3)
        st.active.fill_(True)
        fn = graph.replay

    with torch.no_grad():
        for i in range(warmup):
            st.host_len = ctx + i + 1
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        t0 = time.perf_counter()
        for i in range(steps):
            st.host_len = ctx + warmup + i + 1
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

    itl = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    tokens = st.token_log[:, 0].tolist()
    row = {"method": method, "batch": B, "ctx": ctx, "steps": steps,
           "cuda_graph": method in GRAPHED,
           "prefill_ms_1seq": pre.ms,
           "itl_ms_mean": sum(itl) / len(itl),
           "itl_ms_p50": itl[len(itl) // 2],
           "itl_ms_p99": itl[min(len(itl) - 1, int(math.ceil(0.99 * len(itl))) - 1)],
           "tok_s": B * steps / wall,
           "kv_cache_gb": st.kv_bytes() / 1e9,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    del st
    if method != "sdpa_eager":
        del graph, fn
    return row, tokens


def agree_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def run_matrix(M, model_name, batches, contexts, methods, steps, warmup, block_size, seed=0,
               device="cuda"):
    rows = []
    for ctx in contexts:
        g = torch.Generator().manual_seed(seed + ctx)
        prompt = torch.randint(0, M.vocab, (ctx,), generator=g).to(device)
        gc.collect()
        torch.cuda.empty_cache()
        pre = PrefillCache(M, prompt)  # one 1-sequence prefill per ctx, reused below
        print("prefill ctx={}: {:.1f} ms (1 sequence, timed once)".format(ctx, pre.ms))
        for B in batches:
            ref = None
            for method in methods:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                need = estimate_bytes(M, method, B, ctx, warmup + steps, block_size)
                free, _ = torch.cuda.mem_get_info()
                base = {"model": model_name, "method": method, "batch": B, "ctx": ctx}
                if need > 0.95 * free:
                    print("{:>13} B={:<3} ctx={:<6} skipped: needs ~{:.1f} GB, {:.1f} GB free"
                          .format(method, B, ctx, need / 1e9, free / 1e9))
                    rows.append(dict(base, skipped="does not fit",
                                     need_gb=need / 1e9, free_gb=free / 1e9))
                    continue
                try:
                    row, tokens = run_one(M, method, B, ctx, steps, warmup, block_size, pre,
                                          device)
                except torch.cuda.OutOfMemoryError:
                    print("{:>13} B={:<3} ctx={:<6} OOM".format(method, B, ctx))
                    rows.append(dict(base, skipped="oom"))
                    continue
                row["model"] = model_name
                if method == "sdpa_eager":
                    ref = tokens
                elif ref is not None:
                    row["agree_prefix_vs_sdpa_eager"] = agree_prefix(tokens, ref)
                row["tokens_head"] = tokens[:16]
                rows.append(row)
                print("{:>13} B={:<3} ctx={:<6} ITL {:8.3f} ms (p50 {:.3f}, p99 {:.3f})  "
                      "{:9.1f} tok/s  KV {:6.2f} GB  agree {}".format(
                          method, B, ctx, row["itl_ms_mean"], row["itl_ms_p50"],
                          row["itl_ms_p99"], row["tok_s"], row["kv_cache_gb"],
                          row.get("agree_prefix_vs_sdpa_eager", "-")))
        del pre
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="HF model id (e.g. meta-llama/Llama-3.1-8B) or 'tiny-random'")
    ap.add_argument("--batches", default="1,4,8,16,32,64")
    ap.add_argument("--contexts", default="4096,16384,65536")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--steps", type=int, default=64, help="timed decode steps")
    ap.add_argument("--warmup", type=int, default=8, help="untimed decode steps first")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--label", default="")
    ap.add_argument("--no-save", action="store_true", help="do not write results/ (smoke runs)")
    args = ap.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    methods = [m for m in args.methods.split(",") if m]
    for m in methods:
        if m not in METHODS:
            raise SystemExit("unknown method {} (choose from {})".format(m, METHODS))
    max_len = max(contexts) + args.warmup + args.steps

    hf, overrides = load_model(args.model, max_len)
    M = Model(hf, max_len)
    print("{}: {} layers, Hq {}, Hkv {}, D {}, weights {:.2f} GB{}".format(
        args.model, M.n_layers, M.hq, M.hkv, M.d,
        sum(p.numel() * p.element_size() for p in hf.parameters()) / 1e9,
        ", rope override {}".format(overrides) if overrides else ""))

    rows = run_matrix(M, args.model, batches, contexts, methods, args.steps, args.warmup,
                      args.block_size)
    if not args.no_save:
        extra = {"model": args.model, "rope_overrides": overrides, "steps": args.steps,
                 "warmup": args.warmup, "block_size": args.block_size, "methods": methods,
                 "gpu": torch.cuda.get_device_name(),
                 "metric_notes": "itl_ms_* per decode step from CUDA events; tok_s = B * steps "
                                 "/ wall; prefill_ms_1seq is one sequence (replicated to B)"}
        save_result("e2e_serving", rows,
                    label=args.label or args.model.split("/")[-1], extra=extra)


if __name__ == "__main__":
    main()
