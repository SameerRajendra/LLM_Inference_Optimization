# LLM_Inference_Optimization

**A long-context GQA decode-attention and serving engine in CUDA (`sm_90a`) and Triton for NVIDIA H100 — reaching 86.6% of H100 HBM bandwidth (2,903 GB/s), beating FlashInfer at the kernel level (up to 1.89× on FP8), serving `Qwen/Qwen3-8B` end-to-end with CUDA Graphs at 2.36× over eager PyTorch (2,968.5 tok/s at `B=64` with FP8 KV), scaling `TP=2/4` over NVLink with CUDA Graphs over NCCL (6.5–7.6×), and cutting 64K attention HBM traffic 3.14× via Quest page-sparse decoding.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## The problem

**At long context, the KV cache — not compute — decides how many users a GPU can serve and how fast.** For an 8B GQA model (`32` Q / `8` KV heads, `D=128`), a 64K-token FP16 KV cache costs ~8.4–9.7 GB per sequence depending on layer count, and eager PyTorch decode wastes ~68–80% of step latency on CPU kernel-launch overhead and unfused `RoPE_KV_Append` / `RMSNorm` ops.

This project tackles the decode bottleneck across three levers:
1. **Read bytes efficiently:** Split-KV CUDA (`v4` & `cp.async` Tensor-Core `cuda_tc`) and Triton (`triton_fp16` / `triton_fp8`) kernels, wrapped as `torch.library.custom_op` and replayed in a single CUDA Graph.
2. **Store fewer bytes:** Per-page (`64` tokens × KV head) **FP8 `e4m3`** quantization — halving KV footprint (`39.86 GB → 19.94 GB` at `B=64, 4K`) with `-0.07%` perplexity delta at 8K.
3. **Read fewer bytes at 64K:** Two-stage **Quest query-aware page-sparse decode** using per-page `(K_min, K_max)` bounds — cutting 64K attention bytes to `14–28%` and delivering `1.78–3.14×` attention speedup.

---

## Key Measured Results (NVIDIA H100 80GB HBM3, `sm_90a`)

Full provenance-stamped tables auto-generated from `results/` live in [`RESULTS.md`](RESULTS.md).

### 1. Decode Kernel vs. SDPA & FlashInfer
- **Kernel suite (`B=1,4`, `ctx=4K..64K`):** Triton FP16 hits **`2,903 GB/s` (86.6% HBM peak)** at `B=4, 64K`; CUDA Tensor-Core (`gqa_decode_tc.cu` with `HMMA` + `cp.async` `LDGSTS`, verified via `cuobjdump -sass` in [`results/profiles/h100_sass.md`](results/profiles/h100_sass.md)) cuts `cuda_v4` latency by up to 28–38% on FP8/FP16 (`0.2639 ms` vs `0.4228 ms` at `B=4, 64K` FP8).
- **Head-to-head vs. FlashInfer (`BatchDecodeWithPagedKVCache`, CUDA Graph):** Matches or beats FlashInfer FP16 (`0.74×–0.99×` latency, `max_err = 6.1e-5`) and beats FlashInfer FP8 by **1.48×–2.40×** (`0.42×–0.68×` latency) with per-page scaling.

### 2. End-to-End `Qwen/Qwen3-8B` Serving vs. PyTorch & vLLM

| Engine / Method | B=1, 4K (ms / tok/s) | B=16, 4K (ms / tok/s) | B=32, 4K (ms / tok/s) | B=64, 4K (ms / tok/s) | B=1, 64K (ms / tok/s) |
|---|---:|---:|---:|---:|---:|
| PyTorch `sdpa_eager` | 25.24 ms / 39.6 | 23.11 ms / 692.2 | 22.75 ms / 1,406.4 | 30.64 ms / 2,088.3 | 22.81 ms / 43.8 |
| PyTorch `sdpa_compile` | 14.67 ms / 68.1 | 122.91 ms / 130.2 | 238.34 ms / 134.0 | 467.98 ms / 136.8 | 151.15 ms / 6.6 |
| **`sparse_kv` `paged_fp16` (CUDA Graph)** | **10.68 ms / 93.6** | 14.89 ms / 1,074.5 | 18.47 ms / 1,733.0 | 25.03 ms / 2,556.3 | **13.89 ms / 72.0** |
| **`sparse_kv` `paged_fp8` (CUDA Graph)** | 11.53 ms / 86.7 | 14.97 ms / 1,068.7 | **17.15 ms / 1,865.7** | **21.55 ms / 2,968.5** | 14.27 ms / 70.1 |
| **vLLM `vllm_fp16`** | 8.11 ms / 123.3 | 11.70 ms / 1,367.9 | 15.09 ms / 2,120.1 | - | 11.65 ms / 85.8 |
| **vLLM `vllm_fp8`** | 8.70 ms / 115.0 | 11.51 ms / 1,390.4 | 13.38 ms / 2,391.6 | - | 10.35 ms / 96.7 |

### 3. Quality Gates (`Qwen/Qwen3-8B`) & 64K Quest Sparsity

- **FP8 (`e4m3`) KV cache** (`benchmarks/quality_gate_fp8.py`, 512 teacher-forced steps): `+0.24%` PPL delta at `2K`, **`-0.07%` PPL delta at `8K`**, **`100.0%` confident-step parity**.
- **Quest 64K sparse decode (`B=4, block_size=64`):**
  - At `12.5%` page budget: **`3.14×` (FP16) / `2.78×` (FP8)** attention speedup, `14.1–15.7%` HBM bytes, **fails** quality gate (`+7.66%` PPL delta).
  - At **`25.0%` page budget**: **`2.02×` (FP16) / `1.78×` (FP8)** attention speedup, `26.6–28.2%` HBM bytes, **PASS** quality gate (`+0.19%` PPL delta, `100%` confident-step parity, `3/3` needle recall at 10/50/90% depth).

### 4. Multi-GPU Tensor Parallelism (`TP=2`, `TP=4`) & `nsys` Attribution

- **1-layer `Qwen3-8B` TP block (`NCCL 2.27.3`, CUDA Graph over NCCL):**
  - `TP=2`: eager `0.781 ms` → graph **`0.166 ms`** (**4.71×**) at `B=1, 4K` FP16; `B=64, 16K` FP8 hits **`0.680 ms`** (`25%` faster than FP16 `0.907 ms`).
  - `TP=4`: eager `0.838 ms` → graph **`0.130 ms`** (**6.45×**, up to **7.55×** on FP8 `B=1, 4K`); `B=64, 16K` FP8 hits **`0.461 ms`**.
- **H100 `nsys` breakdown ([`results/profiles/h100_nsys_profile.md`](results/profiles/h100_nsys_profile.md)):** Eager `Decode_Step` spends `43.3–44.0%` in `RoPE_KV_Append` and `23.9–24.8%` in `RMSNorm`, while `cuda_gpu_kern_sum` ([`results/profiles/h100_nsys_kern_sum.txt`](results/profiles/h100_nsys_kern_sum.txt)) shows GPU durations of `1.1–3.7 us` per launch — quantifying why CUDA Graphs eliminate the CPU dispatch wall, and identifying fused `RMSNorm + RoPE + KV Store` as the remaining ~22% gap vs. vLLM.

---

## Repository Layout

```
LLM_Inference_Optimization/
├── csrc/
│   ├── kernels/
│   │   ├── gqa_decode.cu         # Split-KV v3 + v4 (block per KV head) + LSE combine
│   │   ├── gqa_decode_tc.cu      # WP-3 Tensor-Core MMA + cp.async double-buffered decode
│   │   ├── kv_fp8.cu             # Per-page e4m3 FP8 quantization kernels
│   │   ├── sparse_attention.cu   # Historical block-sparse ablation
│   │   └── kv_evict_quant.cu     # Fused top-k eviction / quantization op
│   └── pybind/bindings.cpp
├── sparse_kv/
│   ├── ops.py                    # torch.library.custom_op wrappers (CUDA Graph / compile safe)
│   ├── attention.py
│   └── eviction.py
├── triton_kernels/
│   ├── gqa_decode.py             # Dense split-KV Triton decode (fp16 & FP8)
│   ├── paged_kv.py               # Paged KV cache allocator + RoPE/FP8 append + page bounds
│   ├── paged_gqa_decode.py       # Variable seq_len paged GQA decode + LSE combine
│   └── quest_sparse_decode.py    # WP-4 two-stage Quest page scorer & sparse decode
├── benchmarks/
│   ├── bench_triton_decode.py    # Kernel suite vs SDPA (CUDA v4, cuda_tc, Triton)
│   ├── bench_e2e_serving.py      # Full Qwen3-8B CUDA Graph serving sweep
│   ├── bench_vs_engines.py       # Head-to-head vs FlashInfer & vLLM
│   ├── quality_gate_fp8.py       # Teacher-forced PPL & parity gate for FP8 KV
│   ├── quality_gate_quest.py     # 64K Quest speedup + PPL + needle-in-a-haystack gate
│   ├── bench_tp_decode.py        # TP=2/4 NCCL AllReduce + CUDA Graph benchmark
│   ├── profile_e2e_nsys.py       # NVTX-annotated nsys & CUDA-events breakdown
│   ├── profile_ncu.py            # SASS inspection (cuobjdump) + ncu driver
│   ├── bench_common.py           # Provenance-stamped JSON/Markdown writer
│   └── make_report.py            # Generates RESULTS.md from results/
├── modal_apps/
│   └── h100.py                   # Consolidated Modal H100 runner (--suite kernel|e2e|tp2|tp4|nsys|wp5|all)
├── tests/                        # Unit, Triton-interpret, CUDA, CUDA Graph, & TP tests
├── profiles/                     # WP-5 profiling runbook
├── results/                      # Measured JSON, CSV, and profile artifacts
├── docs/                         # PLAN.md & WORK_PACKAGES.md
├── cluster/                      # Slurm setup & sbatch scripts
└── archive/                      # Archived training script, JAX Pallas ref, & PDF write-up
```

---

## Quickstart

Requires CUDA 12.x (`nvcc` on `PATH`), Python ≥ 3.9, and an NVIDIA GPU (`sm_90a` target).

```bash
git clone https://github.com/SameerRajendra/LLM_Inference_Optimization.git
cd LLM_Inference_Optimization
pip install -r requirements.txt
python setup.py build_ext --inplace
```

Run tests locally (including CPU-only Triton interpreter mode):

```bash
TRITON_INTERPRET=1 pytest tests/ -v   # CPU-only unit + Triton interpreter suite
pytest tests/ -v                      # Full GPU + CUDA extension suite
```

Run H100 suites on Modal ([`modal_apps/h100.py`](modal_apps/h100.py)):

```bash
modal run modal_apps/h100.py --suite kernel   # WP-1/WP-3/WP-4 kernel + quality gates
modal run modal_apps/h100.py --suite e2e      # WP-2 Qwen3-8B vs SDPA, FlashInfer & vLLM
modal run modal_apps/h100.py --suite wp5      # WP-5 TP=2, TP=4, & E2E nsys profiling
```

Regenerate [`RESULTS.md`](RESULTS.md) from JSON/CSV results:

```bash
python benchmarks/make_report.py
```

---

## Completed Milestones & Next Steps

- [x] **WP-0:** Repo hygiene, CI workflow, staged CUDA `v3` → `v4` + FP8 KV optimization.
- [x] **WP-1:** Paged KV pool (`block_size=64`), variable `seq_lens`, and fused RoPE + FP8 KV append.
- [x] **WP-2:** `torch.library.custom_op` + CUDA Graph E2E `Qwen/Qwen3-8B` serving vs. SDPA, FlashInfer, and vLLM.
- [x] **WP-3:** CUDA Tensor-Core MMA + `cp.async` double-buffered kernel (`gqa_decode_tc.cu`) + Triton decode parity.
- [x] **WP-4:** Two-stage Quest 64K page-sparse decode + PPL/needle quality gate (`25%` budget sweet spot).
- [x] **WP-5:** `sm_90a` SASS summary, `nsys` E2E attribution, and `TP=2/4` NCCL + CUDA Graph scaling.
- [ ] **Next:** Fuse `RMSNorm + RoPE + KV Store` into custom Triton/CUDA kernels to close the remaining ~22% E2E gap vs. vLLM, and extend full multi-layer `Qwen3-8B` TP sharding.

---

## References

1. Dao et al., *FlashAttention*, NeurIPS 2022 · FlashDecoding (2023).
2. Ainslie et al., *GQA*, EMNLP 2023.
3. Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP 2023.
4. Tang et al., *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*, ICML 2024.
5. Liu et al., *KIVI: Plug-and-play 2bit KV Cache Quantization*, ICML 2024.

## License

MIT — see [`pyproject.toml`](pyproject.toml).
