# LLM_Inference_Optimization

**A GQA decode-attention kernel for long-context LLM inference on NVIDIA Hopper —
optimised from 5.8% to 56% of H100 HBM peak (9.7× faster) through measured,
staged diagnosis, and benchmarked against FlashAttention at every step.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green?logo=nvidia)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## The problem

**At long context, the KV cache — not compute — decides how many users a GPU can
serve and how fast.** For Llama-3.1-8B the cache costs 128 KiB per token
(32 layers × 8 KV heads × 128 dims × 2 for K/V × 2 bytes). On an 80 GB H100:

| | |
|---|---|
| Model weights (fp16) | 16 GB |
| KV pool left | ~60 GB |
| KV per 64K-token sequence | **8.4 GB** |
| **Concurrent 64K sequences** | **~7** |

Seven users per $30K GPU — and every decode step must re-read all ~59 GB of
their cache. Decode is memory-bandwidth-bound, and the bandwidth goes almost
entirely to the KV cache. This repo attacks that on three fronts: read the
bytes efficiently (done), store fewer bytes (FP8 KV, in progress), and read
fewer of them (paged query-aware sparsity, planned).

## Measured results

H100 SXM (sm_90), Llama-3.1-8B attention shapes (32 Q / 8 KV heads,
head_dim 128), single decode step, batch 1, isolated kernel. Full auditable
history in [`RESULTS.md`](RESULTS.md) — every row pinned to the commit that
produced it.

| Context | Start | fp16 KV | **FP8 KV** | Total speedup | vs PyTorch SDPA (FlashAttention) |
|---:|---:|---:|---:|---:|---|
| 4K | 0.0757 ms | 0.0190 ms | **0.0181 ms** | 4.2× | **0.90× — faster** |
| 16K | 0.4055 ms | 0.0552 ms | **0.0441 ms** | 9.2× | 1.12× slower |
| 64K | 1.3538 ms | 0.1397 ms | **0.1251 ms** | **10.8×** | 1.22× slower |

The fp16 path is numerically unchanged throughout: max abs error vs SDPA is
≤ 0.002 at every context — **one fp16 ULP**, i.e. exact to the last
representable digit. 28 correctness tests gate every performance change,
including partial-tile boundaries (N = 63/64/65) and batching.

FlashAttention sits at ~77% of HBM peak — near the roofline for *exact* dense
attention. **This project does not claim to beat it at that problem.** The
dense kernel's honest target is parity; the wins come from moving fewer bytes.

### FP8 KV cache — quality, measured end to end

e4m3 with one scale per (64-token page, KV head) halves KV traffic *and*
halves resident cache, taking an 80 GB H100 from ~7 to ~14 concurrent 64K
sequences. Validated on **Qwen3-8B** over 512 teacher-forced steps of real
text (not the model's own greedy output, which is near-deterministic by
construction and cannot stress a quantiser):

| Prefill | Perplexity Δ | Parity on confident steps | Raw argmax parity |
|---:|---:|---:|---:|
| 2K | **+0.24%** | **100.0%** | 98.0% |
| 8K | **−0.07%** | **100.0%** | 97.9% |

Raw parity is reported for honesty but is not the criterion: every
disagreement occurred where the reference was ~18–26% confident with a
≤0.03 margin over the runner-up — near-ties, indistinguishable from sampling
noise. **No confident prediction ever changed**, at either context length.
Reproduce with `benchmarks/quality_gate_fp8.py`.

## How it got there

The optimisation is staged so each change has its own measured delta — including
one that didn't work.

| Stage | Change | 64K | Outcome |
|---|---|---:|---|
| 0 | Split-KV baseline (FlashDecoding-style) | 1.3538 ms | 5.8% of peak |
| 1 | Remove a 32-way shared-memory bank conflict | 1.3752 ms | **no effect** |
| 2 | Block per KV head + warp per query head | 0.1697 ms | **8.0×** |
| 3a | Size split count to whole waves | 0.1397 ms | 1.21× |
| WS3 | FP8 (e4m3) KV cache, per-page scales | **0.1251 ms** | 1.12×, 2× fewer bytes |
| 4a | Halve tile to double occupancy | 0.1372 ms | **reverted — slower** |

**Stage 1 was a dud, and it's the most instructive step.** `tile_K` had a
128-half row stride, so the dot-product read `tile_K[tid][d]` mapped to bank
`(tid*64 + d/2) % 32` — and `tid*64 % 32 == 0` for every thread, putting all 32
lanes of a warp on one bank. A real 32-way conflict, correctly diagnosed and
removed. It bought nothing, because at 194 GB/s the kernel was nowhere near
saturating any memory path: it was **latency-starved, not throughput-bound**,
and the conflict had been hiding behind memory stalls the whole time.

That redirected the search to two facts provable from the launch configuration
rather than from a profiler (this cluster has no Nsight Compute counter access):

- **4× redundant traffic** — the grid was indexed by *query* head with
  `kv_head = q_head / 4`, so four blocks fetched identical K/V. At 64K that is
  1.02 GB of requests against 262 MB of unique KV.
- **18.8% occupancy** — 65 KB of shared memory per block allowed only 3 blocks
  per SM, i.e. 12 warps to hide ~500-cycle latency.

**Stage 2** fixed both with one restructure: index the grid by KV head so each
tile is fetched once and shared by all four query heads, and halve the tile to
64 tokens so shared drops to 34.5 KB and 6 blocks fit per SM
(37.5% occupancy — verified via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
not estimated). Giving each warp one query head made the whole online softmax a
warp-shuffle reduction: 2 barriers per tile, down from ~9.

**Stage 3a** was wave quantisation. The grid contributes `Hkv × B` blocks per
split against a 792-block residency (132 SMs × 6); at 64K it launched 1000
blocks — one full wave plus a wave 26% occupied. Choosing the split count from
measured residency, rounded to whole waves, took utilisation to 100%. It
predicted 16K would gain most (33% → 100%), and 16K gained most.

**Stage 4a was the second negative result, and it located the current limit.**
After FP8 halved the bytes, `%HBM` *fell* from 56% to 31% — the kernel had
stopped being bandwidth-bound, and ~86 µs of the 64K runtime no longer depended
on bytes at all. Halving the tile to double occupancy (37.5% → 75%) made
everything slower, which rules occupancy out too. What remains is instruction
issue: ~1700 instructions per lane per tile against 4 issue slots/cycle and 24
warps/SM lands near the observed floor. That is a *count* problem, so the next
step is doing more work per instruction — tensor-core MMA, where one
`m16n8k16` performs 2048 MACs against 32 for a warp of scalar FMAs.

## Skills demonstrated

| Domain | Details |
|---|---|
| **CUDA / Hopper** | Split-KV FlashDecoding-style decode, online-softmax LSE combine, warp-shuffle reductions, shared-memory bank-conflict analysis, vectorised (`uint4`) global loads, occupancy and wave-quantisation tuning |
| **Performance methodology** | Roofline and effective-bandwidth accounting, staged A/B with per-stage records, launch-config analysis without profiler counters, hypothesis → measurement → revision (including a falsified hypothesis) |
| **LLM inference internals** | GQA decode, KV-cache capacity and bandwidth modelling, long-context serving economics |
| **Engineering practice** | Provenance-stamped result recording, correctness gates on tile boundaries, reproducible-from-SHA benchmarks |

## Repository layout

```
LLM_Inference_Optimization/
├── csrc/kernels/
│   ├── gqa_decode.cu           # v3 split-KV + v4 (block per KV head) + LSE combine
│   ├── sparse_attention.cu     # block-sparse decode (ablation)
│   └── kv_evict_quant.cu       # top-k sparse / eviction (ablation)
├── sparse_kv/                  # Python package wrapping sparse_kv._C
├── benchmarks/
│   ├── profile_kernel_vs_sdpa.py   # staged kernel benchmark vs FlashAttention
│   ├── bench_serving_regime.py     # batch × context sweep, KV capacity, tokens/s
│   ├── profile_ncu.py              # single-launch driver for Nsight Compute
│   ├── bench_common.py             # provenance-stamped result recording
│   └── make_report.py              # generates RESULTS.md from results/
├── cluster/                    # Slurm runbook, env setup, sbatch scripts
├── tests/                      # correctness gates (v4 vs SDPA vs v3)
├── docs/PLAN.md                # problem statement and workstream design
└── results/                    # every benchmark run, pinned to its commit
```

## Quickstart

Requires CUDA 12.x (`nvcc` on `PATH`), Python ≥ 3.9, and an NVIDIA GPU
(kernels are built for `sm_90a`).

```bash
git clone https://github.com/SameerRajendra/LLM_Inference_Optimization.git
cd LLM_Inference_Optimization
python -m pip install --user --upgrade setuptools wheel ninja pybind11
pip install --user torch --extra-index-url https://download.pytorch.org/whl/cu124
pip install --user -r requirements.txt
python setup.py build_ext --inplace       # builds sparse_kv._C in place
```

```bash
pytest tests/ -v
python benchmarks/profile_kernel_vs_sdpa.py --kernel v4
python benchmarks/bench_serving_regime.py
```

On a Slurm cluster see [`cluster/CLUSTER.md`](cluster/CLUSTER.md).

## Roadmap

- [x] **FP8 (e4m3) KV cache** with per-page scales, dequantised in-register on
      load — 2× fewer KV bytes at +0.24% perplexity, validated to 8K context.
- [ ] **Tensor-core MMA for QK^T and PV.** The measured bottleneck is now
      instruction issue, not bandwidth or occupancy (both ruled out by
      experiment above). This is the fix that addresses it, and unlike the
      byte-reduction work it needs no quality trade-off: HMMA multiplies in
      fp16 and accumulates in fp32, so numerics are unchanged.
- [ ] **Paged KV + Quest-style query-aware sparsity** — per-page criticality
      metadata, load only the top-k pages. Converts the existing sparse
      *ablations* (which read all of K before selecting, so save nothing) into a
      real bandwidth reduction. Composes with FP8: 8× fewer bytes together.
- [ ] **Measured serving comparison against vLLM** (`--kv-cache-dtype fp8`) at
      matched context, replacing the modelled tokens/sec projections in
      `bench_serving_regime.py` with numbers from a production engine.
- [ ] **Triton port** of the decode kernel for a cross-language comparison.

## References

1. Dao et al., *FlashAttention*, NeurIPS 2022 · FlashDecoding (2023).
2. Ainslie et al., *GQA*, EMNLP 2023.
3. Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP 2023.
4. Tang et al., *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*, ICML 2024.
5. Liu et al., *KIVI: Plug-and-play 2bit KV Cache Quantization*, ICML 2024.

## License

MIT — see [`pyproject.toml`](pyproject.toml).
