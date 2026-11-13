"""
WP-4 speed + quality gate for Quest-style page-sparse decode.

Part 1 -- kernel (any CUDA GPU; RTX 4060 or H100):
  Synthetic paged pool at --seq-len (default 64K) with Qwen3-8B attention geometry
  (32 Q / 8 KV heads, head_dim 128), shuffled block table, fp16 and FP8 KV. For each page
  budget (default 12.5% and 25%) it reports
    * latency of dense paged_gqa_decode vs the full Quest step (scorer + top-k + sparse
      decode + combine), both replayed from CUDA graphs, L2 flushed between reps
      (triton.testing.do_bench), plus an eager per-stage breakdown;
    * effective HBM bytes of both paths (quest_hbm_bytes: K/V of the selected pages,
      metadata, scores/selection, q/out; dense: all K/V), and achieved GB/s;
    * a 100%-budget parity check against dense paged decode on the same pool.
  Random keys make the selection meaningless, which is fine here: latency depends only on
  how many pages are read. Quality is Part 2's job.

Part 2 -- quality (needs ~18 GB for Qwen/Qwen3-8B fp16: H100/A100/L40S, not an 8 GB card):
  Decode attention (S == 1) of every layer after the first --dense-prefix-layers is routed
  through a per-layer paged pool that is appended incrementally (reshape_and_cache with
  k_min/k_max, exactly as a serving engine would), then
    paged   dense paged_gqa_decode   (control: isolates kernel error from sparsity)
    quest   quest_sparse_decode at each budget (fraction of the CURRENT context's pages)
  Prefill is not intercepted, so all paths share an exact prefill.
    * teacher-forced replay of real text (same protocol and metrics as quality_gate_fp8.py):
      argmax parity, confident-step parity, top-5 hit, KL, perplexity;
    * needle-in-a-haystack: a passkey hidden at several depths of a --needle-context prompt;
      the question tokens are fed one at a time through the decode path, so the answer is
      produced by sparse decode attention, then decoded greedily.
  Gate per budget: confident-step parity >= --min-confident-parity, |dPPL| <= --max-ppl-delta-pct,
  and every needle the exact reference retrieves is retrieved.

Run (no Modal; a local GPU or an interactive node):
  python benchmarks/quality_gate_quest.py --parts kernel                 # RTX 4060 / H100
  python benchmarks/quality_gate_quest.py --parts kernel --batch 8 --budgets 0.0625,0.125,0.25,0.5
  python benchmarks/quality_gate_quest.py --parts quality --dense-prefix-layers 2
  python benchmarks/quality_gate_quest.py                                # both

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton                                                    # noqa: E402
from triton.testing import do_bench                              # noqa: E402

from bench_common import save_result                             # noqa: E402
from sparse_kv import ops                                        # noqa: E402
from triton_kernels.paged_gqa_decode import paged_gqa_decode     # noqa: E402
from triton_kernels.paged_kv import (                            # noqa: E402
    allocate_page_minmax, allocate_paged_kv_cache, reshape_and_cache, reshape_and_cache_fp8,
)
from triton_kernels.quest_sparse_decode import (                 # noqa: E402
    pages_for_budget, paged_sparse_decode, quest_hbm_bytes, quest_page_scores,
    quest_select_pages, quest_sparse_decode,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


# ===================================================================== Part 1: kernel bench

def build_synthetic_pool(seq_lens, Hkv, D, block_size, fp8, seed=0):
    """Shuffled block table over a shared pool, filled through the real append path
    (so k_min/k_max come from reshape_and_cache, not from a shortcut)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    B = len(seq_lens)
    pages = [triton.cdiv(s, block_size) for s in seq_lens]
    num_blocks = sum(pages)
    perm = torch.randperm(num_blocks, device="cuda", generator=g).to(torch.int32)
    bt = torch.zeros(B, max(pages), dtype=torch.int32, device="cuda")
    i = 0
    for b, n in enumerate(pages):
        bt[b, :n] = perm[i:i + n]
        i += n
    kc, vc, ks, vs = allocate_paged_kv_cache(num_blocks, block_size, Hkv, D, fp8=fp8)
    k_min, k_max = allocate_page_minmax(num_blocks, Hkv, D)
    chunk = 8192
    for b, s in enumerate(seq_lens):
        for t0 in range(0, s, chunk):
            t1 = min(s, t0 + chunk)
            pos = torch.arange(t0, t1, device="cuda")
            slots = (bt[b, pos // block_size].long() * block_size + pos % block_size).int()
            k = torch.randn(t1 - t0, Hkv, D, device="cuda", generator=g).half()
            v = torch.randn(t1 - t0, Hkv, D, device="cuda", generator=g).half()
            if fp8:
                reshape_and_cache_fp8(k, v, kc, vc, ks, vs, slots, k_min=k_min, k_max=k_max)
            else:
                reshape_and_cache(k, v, kc, vc, slots, k_min=k_min, k_max=k_max)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    return dict(kc=kc, vc=vc, ks=ks, vs=vs, k_min=k_min, k_max=k_max, bt=bt, sl=sl)


def time_graph_ms(fn, rep_ms):
    """Median latency of one CUDA-graph replay of fn, L2 flushed before every rep."""
    graph, _ = ops.capture_cuda_graph(fn)
    return do_bench(graph.replay, warmup=25, rep=rep_ms, return_mode="median")


def bench_kernel(args):
    dev_name = torch.cuda.get_device_name()
    Hq, Hkv, D, B, S = args.q_heads, args.kv_heads, args.head_dim, args.batch, args.seq_len
    if args.ragged:  # lengths spread evenly over [S/2, S]
        seq_lens = [S // 2 + (S - S // 2) * (b + 1) // B for b in range(B)]
    else:
        seq_lens = [S] * B
    print("== kernel: {} | B={} seq_lens={}..{} Hq={} Hkv={} D={} recent_pages={}".format(
        dev_name, B, min(seq_lens), max(seq_lens), Hq, Hkv, D, args.recent_pages))
    hdr = "{:<5} {:>5} {:>7} {:>6} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8} {:>9}".format(
        "kv", "page", "budget", "top_k", "dense us", "quest us", "speedup", "dense MB",
        "quest MB", "bytes %", "quest GB/s")
    rows = []
    for kv in args.kv.split(","):
        fp8 = kv == "fp8"
        for bs in _ints(args.block_sizes):
            if fp8 and bs % 64:
                print("skip fp8 block_size {} (FP8 pool needs a multiple of 64)".format(bs))
                continue
            torch.cuda.empty_cache()
            pool = build_synthetic_pool(seq_lens, Hkv, D, bs, fp8, seed=args.seed)
            q = torch.randn(B, Hq, D, device="cuda").half()
            kv_args = dict(k_scale=pool["ks"], v_scale=pool["vs"])
            max_pages = triton.cdiv(S, bs)

            def dense():
                return ops.paged_gqa_decode(q, pool["kc"], pool["vc"], pool["bt"], pool["sl"],
                                            S, **kv_args)

            t_dense = time_graph_ms(dense, args.rep_ms)

            # Sanity: 100% budget must reproduce dense on this exact pool.
            full = quest_sparse_decode(q, pool["kc"], pool["vc"], pool["bt"], pool["sl"],
                                       pool["k_min"], pool["k_max"], max_pages,
                                       recent_pages=args.recent_pages, max_seq_len=S, **kv_args)
            parity = (full.float() - dense().float()).abs().max().item()
            print("\n[{} page={}] dense {:.1f} us | 100%-budget max|quest-dense| = {:.2e}"
                  .format(kv, bs, t_dense * 1e3, parity))
            print(hdr)
            for budget in _floats(args.budgets):
                top_k = pages_for_budget(budget, S, bs, args.recent_pages)

                def quest(top_k=top_k):
                    return ops.quest_sparse_decode(
                        q, pool["kc"], pool["vc"], pool["bt"], pool["sl"], pool["k_min"],
                        pool["k_max"], top_k, S, recent_pages=args.recent_pages, **kv_args)

                t_quest = time_graph_ms(quest, args.rep_ms)

                # Eager per-stage breakdown (includes launch overhead; sums > graph time).
                scores = quest_page_scores(q, pool["k_min"], pool["k_max"], pool["bt"],
                                           pool["sl"], bs, args.recent_pages, S)
                sel = quest_select_pages(scores, top_k)
                t_score = do_bench(lambda: quest_page_scores(
                    q, pool["k_min"], pool["k_max"], pool["bt"], pool["sl"], bs,
                    args.recent_pages, S), rep=args.rep_ms, return_mode="median")
                t_select = do_bench(lambda: quest_select_pages(scores, top_k), rep=args.rep_ms,
                                    return_mode="median")
                t_sparse = do_bench(lambda: paged_sparse_decode(
                    q, pool["kc"], pool["vc"], pool["bt"], pool["sl"], sel, **kv_args),
                    rep=args.rep_ms, return_mode="median")

                acc = quest_hbm_bytes(sel, pool["sl"], bs, Hq, D, fp8=fp8, max_pages=max_pages)
                row = {
                    "device": dev_name, "kv": kv, "block_size": bs, "batch": B,
                    "seq_lens": seq_lens, "q_heads": Hq, "kv_heads": Hkv, "head_dim": D,
                    "budget": budget, "top_k_pages": top_k, "total_pages_per_seq": max_pages,
                    "recent_pages": args.recent_pages,
                    "dense_us": t_dense * 1e3, "quest_us": t_quest * 1e3,
                    "speedup": t_dense / t_quest,
                    "eager_score_us": t_score * 1e3, "eager_select_us": t_select * 1e3,
                    "eager_sparse_decode_us": t_sparse * 1e3,
                    "dense_gbps": acc["dense_total_bytes"] / (t_dense * 1e-3) / 1e9,
                    "quest_gbps": acc["sparse_total_bytes"] / (t_quest * 1e-3) / 1e9,
                    "full_budget_max_abs_diff": parity,
                    "timing": "CUDA graph replay, median, L2 flushed (triton do_bench)",
                }
                row.update(acc)
                rows.append(row)
                print("{:<5} {:>5} {:>6.1%} {:>6} {:>10.1f} {:>10.1f} {:>7.2f}x {:>10.1f} "
                      "{:>10.1f} {:>7.1f}% {:>9.0f}".format(
                          kv, bs, budget, top_k, row["dense_us"], row["quest_us"],
                          row["speedup"], acc["dense_total_bytes"] / 1e6,
                          acc["sparse_total_bytes"] / 1e6, 100 * acc["bytes_ratio"],
                          row["quest_gbps"]))
                print("      stages (eager): score {:.1f} us, select {:.1f} us, sparse decode "
                      "{:.1f} us".format(t_score * 1e3, t_select * 1e3, t_sparse * 1e3))
            pool = None  # free this pool before the next config allocates its own
    if not args.no_save:
        save_result("quest_kernel", rows, label=args.label or "quest-kernel-{}k".format(S // 1024),
                    extra={"protocol": "synthetic random K/V, shuffled block table; dense = "
                                       "paged_gqa_decode, quest = score+select+sparse decode"})
    return rows


# ==================================================================== Part 2: quality gate

_ORIG_SDPA = F.scaled_dot_product_attention
_PATCHED_MODULES = []
_STATE = {"mode": "ref", "layer": -1, "pools": {}}
_STATS = {"intercepted": 0, "passed_through": 0, "dense_prefix": 0}


class _LayerPool:
    """One layer's paged KV pool for a batch-1 sequence, appended incrementally."""

    def __init__(self, Hkv, D, block_size, capacity, fp8, seed):
        self.bs, self.fp8 = block_size, fp8
        nb = triton.cdiv(capacity, block_size)
        g = torch.Generator().manual_seed(seed)
        self.bt = torch.randperm(nb, generator=g).to(torch.int32).view(1, nb).cuda()
        self.kc, self.vc, self.ks, self.vs = allocate_paged_kv_cache(nb, block_size, Hkv, D,
                                                                     fp8=fp8)
        self.k_min, self.k_max = allocate_page_minmax(nb, Hkv, D)
        self.n = 0

    def sync(self, key, value):
        """Append the tokens of the HF cache [1, Hkv, N, D] this pool has not seen yet."""
        N = key.shape[2]
        assert N >= self.n, "cache shrank; reset the pools between runs"
        if N == self.n:
            return
        pos = torch.arange(self.n, N, device=key.device)
        slots = (self.bt[0, pos // self.bs].long() * self.bs + pos % self.bs).int()
        k = key[0, :, self.n:N].transpose(0, 1).contiguous()
        v = value[0, :, self.n:N].transpose(0, 1).contiguous()
        if self.fp8:
            reshape_and_cache_fp8(k, v, self.kc, self.vc, self.ks, self.vs, slots,
                                  k_min=self.k_min, k_max=self.k_max)
        else:
            reshape_and_cache(k, v, self.kc, self.vc, slots, k_min=self.k_min, k_max=self.k_max)
        self.n = N


def _quest_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                **kwargs):
    B, H, S, D = query.shape
    Hkv = key.shape[1]
    mode = _STATE["mode"]
    ok = (mode != "ref" and S == 1 and B == 1 and D in (64, 128) and attn_mask is None
          and dropout_p == 0.0 and query.dtype == torch.float16 and H % Hkv == 0
          and query.is_cuda and key.shape[2] > 0)
    if not ok:
        _STATS["passed_through"] += 1
        return _ORIG_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, **kwargs)
    layer = _STATE["layer"]
    if layer < _STATE["dense_prefix_layers"]:
        _STATS["dense_prefix"] += 1
        return _ORIG_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, **kwargs)

    _STATS["intercepted"] += 1
    pool = _STATE["pools"].get(layer)
    if pool is None:
        pool = _LayerPool(Hkv, D, _STATE["block_size"], _STATE["capacity"], _STATE["fp8"],
                          seed=layer)
        _STATE["pools"][layer] = pool
    pool.sync(key, value)
    N = key.shape[2]
    sl = torch.tensor([N], dtype=torch.int32, device=query.device)
    q = query[:, :, 0, :].contiguous()
    sc = scale if scale is not None else 1.0 / math.sqrt(D)
    kv = dict(k_scale=pool.ks, v_scale=pool.vs)
    if mode == "paged":
        out = paged_gqa_decode(q, pool.kc, pool.vc, pool.bt, sl, scale=sc, max_seq_len=N, **kv)
    else:
        bs, recent = _STATE["block_size"], _STATE["recent_pages"]
        top_k = pages_for_budget(_STATE["budget"], N, bs, recent)
        out = quest_sparse_decode(q, pool.kc, pool.vc, pool.bt, sl, pool.k_min, pool.k_max,
                                  top_k, recent_pages=recent, scale=sc, max_seq_len=N, **kv)
    return out.unsqueeze(2)


def install_sdpa():
    F.scaled_dot_product_attention = _quest_sdpa
    torch.nn.functional.scaled_dot_product_attention = _quest_sdpa
    # transformers may have bound SDPA at import time; rebind any module still holding it.
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("transformers"):
            continue
        try:
            if getattr(mod, "scaled_dot_product_attention", None) is _ORIG_SDPA:
                setattr(mod, "scaled_dot_product_attention", _quest_sdpa)
                _PATCHED_MODULES.append(mod)
        except Exception:
            continue


def restore_sdpa():
    F.scaled_dot_product_attention = _ORIG_SDPA
    torch.nn.functional.scaled_dot_product_attention = _ORIG_SDPA
    for mod in _PATCHED_MODULES:
        try:
            setattr(mod, "scaled_dot_product_attention", _ORIG_SDPA)
        except Exception:
            continue
    del _PATCHED_MODULES[:]


def set_mode(mode, budget=None):
    _STATE["mode"], _STATE["budget"] = mode, budget
    _STATE["pools"] = {}
    for k in _STATS:
        _STATS[k] = 0
    torch.cuda.empty_cache()


def register_layer_hooks(model):
    """Record which decoder layer is running, so the first N layers can stay dense."""
    layers = model.model.layers
    for i, layer in enumerate(layers):
        def hook(mod, args, i=i):
            _STATE["layer"] = i
        layer.self_attn.register_forward_pre_hook(hook)
    return len(layers)


def load_corpus(text_file):
    """Varied real prose (this repo's docs by default) so perplexity is not degenerate."""
    if text_file:
        with open(text_file, encoding="utf-8") as f:
            return f.read()
    parts = []
    for rel in ("README.md", "docs/PLAN.md", "docs/WORK_PACKAGES.md", "cluster/CLUSTER.md",
                "RESULTS.md", "csrc/kernels/gqa_decode.cu", "triton_kernels/paged_kv.py"):
        p = os.path.join(REPO_ROOT, rel)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="ignore") as f:
                parts.append(f.read())
    return "\n\n".join(parts)


@torch.no_grad()
def run_teacher_forced(model, ids, cont):
    """Prefill ids, then feed the real continuation one token at a time; logits[i] predicts
    cont[i] and comes from one decode step."""
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :].float()
    all_logits = []
    for t in cont:
        all_logits.append(logits.squeeze(0).clone())
        out = model(torch.tensor([[int(t)]], device=ids.device), past_key_values=cache,
                    use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :].float()
    return torch.stack(all_logits)


@torch.no_grad()
def run_needle(model, tok, prefix, tail, new_tokens):
    """Prefill prefix (exact), push the question tail through the decode path token by token,
    then decode the answer greedily: every answer logit comes from decode attention."""
    out = model(prefix, use_cache=True)
    cache = out.past_key_values
    for t in tail:
        out = model(torch.tensor([[int(t)]], device=prefix.device), past_key_values=cache,
                    use_cache=True)
        cache = out.past_key_values
    gen = []
    for _ in range(new_tokens):
        nxt = int(out.logits[0, -1].argmax())
        gen.append(nxt)
        out = model(torch.tensor([[nxt]], device=prefix.device), past_key_values=cache,
                    use_cache=True)
        cache = out.past_key_values
    return tok.decode(gen)


def needle_cases(tok, corpus_ids, ctx, depths, seed):
    rng = random.Random(seed)
    cases = []
    for depth in depths:
        key = str(rng.randint(10000, 99999))
        needle = tok(" The secret passkey is {}. Remember it: the secret passkey is {}. "
                     .format(key, key), add_special_tokens=False).input_ids
        question = tok("\n\nWhat is the secret passkey mentioned in the text above? "
                       "The secret passkey is", add_special_tokens=False).input_ids
        n_fill = ctx - len(needle) - len(question)
        reps = max(1, -(-n_fill // max(len(corpus_ids), 1)))
        filler = (corpus_ids * reps)[:n_fill]
        at = int(depth * n_fill)
        prefix = filler[:at] + needle + filler[at:]
        cases.append({"depth": depth, "key": key, "prefix": prefix, "tail": question})
    return cases


def score_logits(ref, test, cont):
    ref_am, test_am = ref.argmax(-1), test.argmax(-1)
    agree = ref_am == test_am
    p_ref, p_test = F.log_softmax(ref, -1), F.log_softmax(test, -1)
    tgt = torch.tensor(cont, device=ref.device).unsqueeze(1)
    ppl_ref = math.exp(-p_ref.gather(1, tgt).mean().item())
    ppl_test = math.exp(-p_test.gather(1, tgt).mean().item())
    conf = p_ref.exp().max(-1).values
    confident = conf >= 0.9
    return {
        "argmax_parity": agree.float().mean().item(),
        "parity_confident_steps": (agree[confident].float().mean().item()
                                   if confident.any() else float("nan")),
        "top5_hit": (test.topk(5, -1).indices == ref_am.unsqueeze(-1)).any(-1)
        .float().mean().item(),
        "kl_ref_test": F.kl_div(p_test, p_ref, log_target=True, reduction="batchmean").item(),
        "ppl_ref": ppl_ref, "ppl": ppl_test,
        "ppl_delta_pct": 100 * (ppl_test - ppl_ref) / ppl_ref,
        "ref_top1_confidence": conf.mean().item(),
    }


def bench_quality(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("== quality: loading {} (fp16, sdpa attention)...".format(args.model))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa")
    model = model.to("cuda").eval()
    n_layers = register_layer_hooks(model)
    cfg = model.config
    fp8 = args.quality_kv == "fp8"
    if fp8 and args.quality_block_size % 64:
        raise SystemExit("--quality-kv fp8 needs --quality-block-size multiple of 64")
    _STATE.update(dense_prefix_layers=args.dense_prefix_layers,
                  block_size=args.quality_block_size, recent_pages=args.recent_pages, fp8=fp8,
                  capacity=max(args.prefill + args.steps, args.needle_context) + 64)
    print("geometry: {} Q / {} KV heads, {} layers; dense prefix layers {}, page {} tokens, "
          "{} KV, recent_pages {}".format(cfg.num_attention_heads, cfg.num_key_value_heads,
                                          n_layers, args.dense_prefix_layers,
                                          args.quality_block_size, args.quality_kv,
                                          args.recent_pages))

    all_ids = tok(load_corpus(args.text_file), return_tensors="pt").input_ids[0]
    need = args.prefill + args.steps
    if all_ids.numel() < need:
        raise SystemExit("corpus gives {} tokens, need {}; use --text-file"
                         .format(all_ids.numel(), need))
    ids = all_ids[:args.prefill].unsqueeze(0).cuda()
    cont = all_ids[args.prefill:need].tolist()
    cases = needle_cases(tok, all_ids.tolist(), args.needle_context,
                         _floats(args.needle_depths), args.seed) if args.needle_context else []

    budgets = _floats(args.budgets)
    modes = [("ref", None), ("paged", None)] + [("quest", b) for b in budgets]
    expected = (n_layers - args.dense_prefix_layers) * args.steps
    install_sdpa()
    results = {}
    try:
        for mode, budget in modes:
            name = mode if budget is None else "quest@{:g}".format(budget)
            set_mode(mode, budget)
            logits = run_teacher_forced(model, ids, cont)
            stats = dict(_STATS)
            if mode != "ref" and stats["intercepted"] < 0.9 * expected:
                raise SystemExit("ERROR: {} handled only {} of ~{} decode attentions ({} fell "
                                 "back) - this run would measure nothing".format(
                                     name, stats["intercepted"], expected,
                                     stats["passed_through"]))
            needles = []
            for c in cases:
                set_mode(mode, budget)
                text = run_needle(model, tok, torch.tensor([c["prefix"]], device="cuda"),
                                  c["tail"], args.needle_new_tokens)
                needles.append({"depth": c["depth"], "found": c["key"] in text,
                                "answer": text.strip()})
            results[name] = {"logits": logits, "stats": stats, "needles": needles,
                             "mode": mode, "budget": budget}
            print("  {:<12} intercepted {:>5}, dense-prefix {:>4}, passthrough {:>4}; needles {}"
                  .format(name, stats["intercepted"], stats["dense_prefix"],
                          stats["passed_through"],
                          "".join("Y" if n["found"] else "n" for n in needles) or "-"))
    finally:
        restore_sdpa()

    ref = results["ref"]
    ref_found = [n["found"] for n in ref["needles"]]
    rows = []
    print("\n{:<12} {:>8} {:>10} {:>8} {:>9} {:>9} {:>8} {:>8}  {}".format(
        "path", "parity", "conf-par", "top5", "KL", "PPL", "dPPL%", "needle", "verdict"))
    for name, r in results.items():
        if name == "ref":
            continue
        m = score_logits(ref["logits"], r["logits"], cont)
        found = [n["found"] for n in r["needles"]]
        needle_ok = all(f for f, rf in zip(found, ref_found) if rf)
        if m["ref_top1_confidence"] > 0.98:
            verdict = "INCONCLUSIVE"
        elif (m["parity_confident_steps"] >= args.min_confident_parity
              and abs(m["ppl_delta_pct"]) <= args.max_ppl_delta_pct and needle_ok):
            verdict = "PASS"
        else:
            verdict = "FAIL"
        row = dict(m)
        row.update({"path": name, "budget": r["budget"], "verdict": verdict,
                    "needles": r["needles"], "needle_ok": needle_ok,
                    "needle_found": sum(found), "needle_ref_found": sum(ref_found),
                    "intercepted": r["stats"]["intercepted"],
                    "dense_prefix_calls": r["stats"]["dense_prefix"],
                    "passed_through": r["stats"]["passed_through"]})
        rows.append(row)
        print("{:<12} {:>7.1%} {:>9.1%} {:>7.1%} {:>9.5f} {:>9.3f} {:>+7.2f}% {:>4}/{:<3}  {}"
              .format(name, m["argmax_parity"], m["parity_confident_steps"], m["top5_hit"],
                      m["kl_ref_test"], m["ppl"], m["ppl_delta_pct"], sum(found),
                      sum(ref_found), verdict))
    print("reference PPL {:.3f}; reference needles {}/{}; gate: confident-step parity >= {:.1%},"
          " |dPPL| <= {:.2f}%, every reference-retrieved needle retrieved".format(
              rows[0]["ppl_ref"] if rows else float("nan"), sum(ref_found), len(ref_found),
              args.min_confident_parity, args.max_ppl_delta_pct))

    if not args.no_save:
        save_result("quality_quest", rows, label=args.label or "quest-quality-gate", extra={
            "model": args.model, "prefill": args.prefill, "steps": args.steps,
            "dense_prefix_layers": args.dense_prefix_layers,
            "block_size": args.quality_block_size, "kv": args.quality_kv,
            "recent_pages": args.recent_pages, "needle_context": args.needle_context,
            "reference_needles": ref["needles"],
            "protocol": "teacher-forced replay of real text; prefill exact and shared; decode "
                        "attention of layers >= dense_prefix_layers via incrementally appended "
                        "paged pool; budget = fraction of current context pages"})
    return all(r["verdict"] == "PASS" for r in rows if r["path"] != "paged")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--parts", default="kernel,quality", help="kernel, quality, or both")
    ap.add_argument("--budgets", default="0.125,0.25", help="page budgets (fractions)")
    ap.add_argument("--recent-pages", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--no-save", action="store_true", help="print only, write no results JSON")
    k = ap.add_argument_group("kernel")
    k.add_argument("--seq-len", type=int, default=64 * 1024)
    k.add_argument("--batch", type=int, default=4)
    k.add_argument("--ragged", action="store_true", help="seq_lens spread over [S/2, S]")
    k.add_argument("--q-heads", type=int, default=32)
    k.add_argument("--kv-heads", type=int, default=8)
    k.add_argument("--head-dim", type=int, default=128)
    k.add_argument("--kv", default="fp16,fp8")
    k.add_argument("--block-sizes", default="16,64", help="page sizes (fp8 skips non-64x)")
    k.add_argument("--rep-ms", type=int, default=200, help="do_bench measurement window")
    m = ap.add_argument_group("quality")
    m.add_argument("--model", default="Qwen/Qwen3-8B")
    m.add_argument("--dense-prefix-layers", type=int, default=2)
    m.add_argument("--quality-block-size", type=int, default=16)
    m.add_argument("--quality-kv", default="fp16", choices=("fp16", "fp8"))
    m.add_argument("--prefill", type=int, default=4096)
    m.add_argument("--steps", type=int, default=64)
    m.add_argument("--text-file", default="")
    m.add_argument("--needle-context", type=int, default=8192, help="0 disables the needle test")
    m.add_argument("--needle-depths", default="0.1,0.5,0.9")
    m.add_argument("--needle-new-tokens", type=int, default=8)
    m.add_argument("--min-confident-parity", type=float, default=0.99)
    m.add_argument("--max-ppl-delta-pct", type=float, default=2.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU (use tests/test_quest_sparse.py for CPU checks)")
    parts = [p.strip() for p in args.parts.split(",")]
    ok = True
    if "kernel" in parts:
        bench_kernel(args)
    if "quality" in parts:
        ok = bench_quality(args)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
