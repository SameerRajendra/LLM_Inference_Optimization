# Decode-Engine Execution Plan — WS1–WS4

## The problem this project solves

**Long-context LLM decoding is memory-bandwidth-bound, and almost all of that
bandwidth goes to re-reading the KV cache.** Every generated token must stream
the entire KV cache from HBM: for Llama-3.1-8B at 64K context that is
2 × 65,536 tokens × 8 KV heads × 128 dims × 2 bytes = **268 MB per layer, ≈8.6 GB
per token across 32 layers** — a ≥2.6 ms hard floor per token on an H100 SXM
(3.35 TB/s peak) before a single FLOP of anything else. Compute is nowhere near
the limit; **bytes moved per token is the limit.**

This project builds and measures a decode engine that attacks that limit from
three compounding directions, each gated by an exactness/quality test:

1. **Fetch efficiency** (WS1): stream the KV bytes you *do* read at near-roofline
   bandwidth instead of the current 6% of peak.
2. **Bytes per token** (WS3): store KV in FP8 with per-page scales — half the
   bytes for near-zero quality loss.
3. **Tokens read at all** (WS4): paged, query-aware sparsity — load only the
   pages that matter to the current query.

WS2 (Triton) delivers the same kernels in the ecosystem's second language,
proving the designs are portable and comparable.

**Success is defined by measurement**, exactly like the existing README: nothing
is claimed until it reproduces on the H100 via `cluster/slurm/bench.sbatch`.

## Baseline (already measured, on `main`)

| Metric @ 64K context, batch=1 | Value |
|---|---|
| Split-KV GQA decode kernel, isolated latency | 1.359 ms (~6% of HBM peak) |
| PyTorch SDPA / FlashAttention | 0.101 ms (~77% of peak) |
| Exactness vs SDPA, all 32 layers | max abs err ≤ 1e-3 |

The 6%→77% gap is precisely the absence of an asynchronous memory pipeline —
which is WS1.

## Model decision

- **Primary: `meta-llama/Llama-3.1-8B`** — keeps every existing measured number
  comparable; the kernel geometry (32 Q / 8 KV heads, head_dim 128, GQA ratio 4)
  is unchanged. Gated: needs `HF_TOKEN` on the cluster.
- **Secondary: `Qwen/Qwen3-8B`** — 2025-generation, ungated, and the *same*
  per-layer attention geometry (32 Q / 8 KV heads, head_dim 128, 36 layers).
  Its QK-norm sits upstream of attention, so all four workstreams run
  unmodified. It demonstrates the kernels generalize across model families and
  removes gating friction.

Benchmarks take `MODEL_ID`; switching primaries later is a flag, not a rewrite.

## Workstreams

All four live on one branch (`decode-engine`) as independent modules — they
touch disjoint files and can be built/tested in any order on the cluster.

### WS1 — Async memory pipeline for the dense decode kernel

*The core kernel-engineering deliverable.*

- **Design:** rebuild `gqa_decode.cu`'s inner loop as a producer/consumer
  pipeline: producer warps issue `cp.async` (upgrade path: TMA /
  `cp.async.bulk` via CuTe) into double-buffered shared-memory K/V tiles;
  consumer warps run QK^T → online softmax → PV on the previous tile. Memory
  latency is hidden behind compute instead of serialized with it.
- **Tensor cores:** pack the 4 query heads of a GQA group as the M-dimension of
  `mma.sync` tiles (padded 4→16). Expectation set honestly: the *pipeline* is
  the bandwidth lever; MMA is measured and reported either way — decode is
  bandwidth-bound, and knowing that is part of the story.
- **Files:** `csrc/kernels/gqa_decode_async.cu` (new; old kernel kept for A/B),
  binding `fused_gqa_async`, tests, bench flag.
- **Acceptance:** max abs err ≤ 1e-3 vs SDPA; **≥40% of HBM peak at 64K
  (~≤0.20 ms), stretch ≥60%**; Nsight before/after traces.

### WS2 — Triton ports

- **Design:** same split-KV online-softmax algorithm in Triton: grid =
  (kv_head, split), `tl.dot` for QK^T/PV, exact log-sum-exp combine as a second
  kernel (reusing the CUDA combine via the same semantics). No build step —
  pure Python module.
- **Files:** `triton_kernels/gqa_decode_triton.py`, tests, three-way bench
  (CUDA vs Triton vs SDPA).
- **Acceptance:** ≤1e-3 vs SDPA; within ~1.5× of the CUDA kernel's latency;
  published comparison table.

### WS3 — FP8 KV cache with fused dequant

- **Design:** store K/V as `float8_e4m3` with one fp16 scale per **16-token
  page** per head (page size shared with WS4 by design); quantization kernel
  runs at append time; the decode kernel dequantizes *in registers after the
  async copy* — HBM traffic is the fp8 bytes, which is the entire point.
  Extends the existing INT8 accuracy-gate work (`benchmarks/validate_int8_kv.py`).
- **Files:** `csrc/kernels/kv_cache_fp8.cu`, `sparse_kv/kv_cache.py`,
  `benchmarks/validate_fp8_kv.py`, Triton variant in `triton_kernels/`.
- **Acceptance:** ~2× reduction in measured KV bytes; **≥1.5× kernel speedup at
  64K over the fp16 dense path** (on top of WS1); argmax parity ≥99% on a fixed
  eval set and perplexity delta reported.

### WS4 — Paged KV + Quest-style query-aware sparsity

- **Design:** block the KV cache into 16-token pages with a page table
  (vLLM-style layout); keep per-page metadata (channelwise min/max of keys);
  per decode step score each page's criticality against the current query and
  **load only the top-k pages**. This converts the existing sparse *ablation*
  (which read everything, then selected) into a real bandwidth reduction, and
  wires in the measured layer-sensitivity result (first N layers stay dense).
- **Files:** `csrc/kernels/paged_gqa_decode.cu`, `sparse_kv/paging.py`, tests,
  bench + quality gate.
- **Acceptance:** bytes moved ∝ page budget (verified by bandwidth accounting);
  **measurable end-to-end attention speedup at 64K with a ≤25% budget**; argmax
  parity ≥99% with the router's dense-layer prefix.

### Composition

WS1's pipeline is the substrate: WS3 changes *what* the pipeline copies (fp8 +
scales), WS4 changes *which pages* it copies. Shared 16-token page abstraction
keeps them composable into the adaptive engine the README's roadmap promises.

## Workflow

Author locally → commit to `decode-engine` → push → `git pull` on cluster →
`sbatch cluster/slurm/{build,test,bench}.sbatch` (see `cluster/CLUSTER.md`).
No code executes locally. Measured results (`results/*.csv|json`) are committed
from the cluster and drive README updates. Nothing lands on `main` or the
README until it reproduces on the H100.

## Showcase — Google XYZ bullets (targets; `[…]` filled only with measured numbers)

1. Raised H100 HBM utilization of a custom GQA decode kernel **from 6% to
   [Y]% of peak** ([1.36 ms → Y ms] at 64K context), by rebuilding it as a
   warp-specialized `cp.async`/TMA double-buffered pipeline with tensor-core
   MMA over GQA query groups.
2. Cut long-context KV-cache traffic **2× with [Y] perplexity delta**, by
   designing an FP8 (e4m3) paged KV cache with per-page scales and
   dequantization fused into the decode kernel.
3. Achieved **[Y]× decode-attention speedup at 64K context with ≥99% argmax
   parity**, by implementing a Quest-style paged sparse kernel that loads only
   the top-k critical KV pages per query.
4. Shipped the same decode kernel in **both CUDA and Triton within [Y]× of
   each other**, benchmarked three-way against FlashAttention with roofline
   accounting.

Job-description coverage added by WS1–WS4: Triton, tensor cores + TMA + warp
specialization on Hopper, FP8 quantization kernels, PagedAttention-style KV
systems, CUTLASS/CuTe — on top of the already-demonstrated CUDA authorship and
Nsight/roofline work.
