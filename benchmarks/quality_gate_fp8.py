"""
End-to-end quality gate for the FP8 KV cache, on a real model and real text.

Why this exists
---------------
tests/test_kv_fp8.py shows cosine ~0.9993 against the fp16 kernel on RANDOM
tensors. That proves the arithmetic is sound. It does not prove that generated
text survives: random tensors have none of the outlier structure real attention
activations have, and the metric that actually matters is whether the sampled
token changes.

Protocol (teacher-forced parity, the standard comparison in the KV-quantisation
literature): generate a reference continuation greedily with exact attention,
then replay THAT SAME token sequence through the fp8 path and compare the
per-step distributions. Replaying rather than generating independently is
deliberate -- two free-running generations diverge after any single
disagreement, which measures chaos, not quality.

Reported per step:
  argmax parity   fraction of steps whose top-1 token is unchanged  <- the gate
  top5 hit        reference top-1 still inside the fp8 top-5
  KL              KL(reference || fp8) over the full vocabulary
  perplexity      of the reference continuation under each path

Prefill is NOT intercepted (S > 1), so both paths share an identical prefill and
every difference measured comes from the fp8 decode path alone.

Run:
  python benchmarks/quality_gate_fp8.py                      # Qwen3-8B, ungated
  python benchmarks/quality_gate_fp8.py --model meta-llama/Llama-3.1-8B
  python benchmarks/quality_gate_fp8.py --prefill 4096 --steps 128

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_kv._C as _C                # noqa: E402
from bench_common import save_result     # noqa: E402

_ORIG_SDPA = F.scaled_dot_product_attention

# Set by install_fp8_sdpa(); counts how often we actually took over, so a run
# that silently never used the kernel cannot be mistaken for a passing gate.
_STATS = {"intercepted": 0, "passed_through": 0}


def _fp8_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
              is_causal=False, scale=None, **kwargs):
    """Substitute the fp8 decode kernel for single-token attention.

    Falls through to real SDPA for prefill (S > 1), for masked calls, and for
    any geometry the kernel does not support -- the fallback is silent to the
    model but counted, and the counts are asserted at the end of the run.
    """
    B, H, S, D = query.shape
    Hkv = key.shape[1]
    ok = (S == 1 and D == 128 and attn_mask is None and dropout_p == 0.0
          and query.dtype == torch.float16 and H % Hkv == 0 and (H // Hkv) <= 32
          and query.is_cuda and key.shape[2] > 0)
    if not ok:
        _STATS["passed_through"] += 1
        return _ORIG_SDPA(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal,
                          scale=scale, **kwargs)

    _STATS["intercepted"] += 1
    sc = scale if scale is not None else (1.0 / math.sqrt(D))
    Q = query.squeeze(2).contiguous()                 # [B, H, D]
    K = key.transpose(1, 2).contiguous()              # [B, N, Hkv, D]
    V = value.transpose(1, 2).contiguous()
    # Re-quantising the whole cache every step is O(N) and slow; a serving
    # engine would quantise once at append time. This harness measures QUALITY,
    # not latency, so correctness beats cleverness here.
    Kq, Vq, ks, vs = _C.quantize_kv_fp8(K, V)
    out = _C.fused_gqa_v4_fp8(Q, Kq, Vq, ks, vs, sc)  # [B, H, D]
    return out.unsqueeze(2)                           # [B, H, 1, D]


_PATCHED_MODULES = []


def install_fp8_sdpa():
    F.scaled_dot_product_attention = _fp8_sdpa
    torch.nn.functional.scaled_dot_product_attention = _fp8_sdpa
    # transformers' sdpa backend may have bound the function at import time
    # (`from torch.nn.functional import scaled_dot_product_attention`), in which
    # case rebinding the module attribute above never reaches the caller and
    # every decode would silently fall through to exact SDPA. Rebind any module
    # still holding the original.
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("transformers"):
            continue
        try:
            if getattr(mod, "scaled_dot_product_attention", None) is _ORIG_SDPA:
                setattr(mod, "scaled_dot_product_attention", _fp8_sdpa)
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


DEFAULT_TEXT = (
    "The design of high-performance inference systems is governed less by "
    "arithmetic throughput than by the movement of bytes. A modern accelerator "
    "can perform hundreds of teraflops, yet spends most of a decode step "
    "waiting on memory. Every generated token requires re-reading the entire "
    "key-value cache, whose size grows linearly with the length of the "
    "conversation. As context windows lengthen, this cache comes to dominate "
    "both the memory capacity of the device and the bandwidth available to it, "
    "and the practical limit on how many users a single accelerator can serve "
    "is set by that cache rather than by any measure of computation. "
)


@torch.no_grad()
def run_reference(model, ids, steps):
    """Greedy decode with exact attention; returns (tokens, logits per step)."""
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :].float()
    toks, all_logits = [], []
    for _ in range(steps):
        nxt = logits.argmax(-1, keepdim=True)
        toks.append(nxt.item())
        all_logits.append(logits.squeeze(0).clone())
        out = model(nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :].float()
    return toks, torch.stack(all_logits)


@torch.no_grad()
def run_replay(model, ids, tokens):
    """Feed a fixed token sequence; returns the logits each step produced."""
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :].float()
    all_logits = []
    for t in tokens:
        all_logits.append(logits.squeeze(0).clone())
        nxt = torch.tensor([[t]], device=ids.device)
        out = model(nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :].float()
    return torch.stack(all_logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="ungated by default; Llama-3.1-8B needs HF_TOKEN")
    ap.add_argument("--prefill", type=int, default=2048,
                    help="approximate prompt length in tokens")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading {} (fp16, sdpa attention)...".format(args.model))
    tok = AutoTokenizer.from_pretrained(args.model)
    # Deliberately no device_map: that routes through accelerate, which on this
    # cluster drags in boto3/botocore and dies on a urllib3 2.x incompatibility.
    # An 8B fp16 model loads fine with a plain .to("cuda").
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = model.to("cuda").eval()

    cfg = model.config
    D = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    print("geometry: {} Q heads / {} KV heads, head_dim {}, {} layers".format(
        cfg.num_attention_heads, cfg.num_key_value_heads, D,
        cfg.num_hidden_layers))
    if D != 128:
        raise SystemExit("kernel requires head_dim=128; this model has {}".format(D))

    text = DEFAULT_TEXT * (args.prefill // 60 + 2)
    ids = tok(text, return_tensors="pt").input_ids[:, :args.prefill].to("cuda")
    print("prefill {} tokens, decoding {} steps\n".format(ids.shape[1], args.steps))

    restore_sdpa()
    ref_toks, ref_logits = run_reference(model, ids, args.steps)

    install_fp8_sdpa()
    _STATS["intercepted"] = 0
    _STATS["passed_through"] = 0
    try:
        fp8_logits = run_replay(model, ids, ref_toks)
    finally:
        restore_sdpa()

    expected = cfg.num_hidden_layers * args.steps
    print("fp8 kernel handled {} decode attentions ({} fell back to SDPA); "
          "expected ~{}".format(_STATS["intercepted"], _STATS["passed_through"],
                                expected))
    if _STATS["intercepted"] < expected * 0.9:
        raise SystemExit("ERROR: the fp8 kernel was barely used - this run "
                         "would measure nothing. Check dtype/head_dim/masking.")

    ref_am = ref_logits.argmax(-1)
    fp8_am = fp8_logits.argmax(-1)
    parity = (ref_am == fp8_am).float().mean().item()

    top5 = fp8_logits.topk(5, dim=-1).indices
    top5_hit = (top5 == ref_am.unsqueeze(-1)).any(-1).float().mean().item()

    p_ref = F.log_softmax(ref_logits, dim=-1)
    p_fp8 = F.log_softmax(fp8_logits, dim=-1)
    kl = F.kl_div(p_fp8, p_ref, log_target=True, reduction="batchmean").item()

    tgt = torch.tensor(ref_toks, device=ref_logits.device)
    nll_ref = -p_ref.gather(1, tgt.unsqueeze(1)).mean().item()
    nll_fp8 = -p_fp8.gather(1, tgt.unsqueeze(1)).mean().item()
    ppl_ref, ppl_fp8 = math.exp(nll_ref), math.exp(nll_fp8)

    print("\n{:<28} {:>12} {:>12}".format("metric", "fp16 (ref)", "fp8 KV"))
    print("-" * 54)
    print("{:<28} {:>12} {:>11.1f}%".format("argmax parity", "100.0%", parity * 100))
    print("{:<28} {:>12} {:>11.1f}%".format("ref top-1 in top-5", "100.0%", top5_hit * 100))
    print("{:<28} {:>12} {:>12.5f}".format("KL(ref || fp8)", "0.00000", kl))
    print("{:<28} {:>12.3f} {:>12.3f}".format("perplexity of continuation",
                                              ppl_ref, ppl_fp8))
    print("-" * 54)
    print("delta perplexity: {:+.3f} ({:+.2f}%)".format(
        ppl_fp8 - ppl_ref, 100 * (ppl_fp8 - ppl_ref) / ppl_ref))

    verdict = "PASS" if (parity >= 0.99 and kl < 0.01) else "FAIL"
    print("\ngate (parity >= 99% and KL < 0.01): {}".format(verdict))

    save_result("quality_fp8", [{
        "model": args.model, "prefill": int(ids.shape[1]), "steps": args.steps,
        "argmax_parity": parity, "top5_hit": top5_hit, "kl_ref_fp8": kl,
        "ppl_ref": ppl_ref, "ppl_fp8": ppl_fp8,
        "ppl_delta_pct": 100 * (ppl_fp8 - ppl_ref) / ppl_ref,
        "intercepted": _STATS["intercepted"],
        "passed_through": _STATS["passed_through"],
        "verdict": verdict,
    }], label=args.label or "fp8-quality-gate",
        extra={"protocol": "teacher-forced replay of the fp16 greedy "
                           "continuation; prefill shared and exact"})
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
