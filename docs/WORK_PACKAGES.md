# High-ROI Work Packages (Offload Spec for Claude Code)

> **Instructions for Claude Code:**
> Execute **only** the Work Package (`WP-X`) requested by the user. Read only the files listed under **"Files to read"** for that package. Do not scan `results/`, `profiles/`, or unrelated directories. After implementing, run the exact commands in **"Verification"**.

---

## WP-0: Repo Hygiene, Bug Fixes & Honest Baseline

**Goal:** Eliminate technical red flags and dead/contradicting code before adding new features.

**Files to read:**
- `sparse_kv/eviction.py`
- `csrc/pybind/bindings.cpp`
- `benchmarks/bench_serving_regime.py`
- `benchmarks/make_report.py`

**Tasks:**
1. **Fix argument mismatch bug in `sparse_kv/eviction.py`:**
   - `csrc/pybind/bindings.cpp` exposes `kv_evict_quant_forward(Q, K, V, top_k, use_int8)` (5 args), whereas `sparse_kv/eviction.py:26` calls `_cuda_evict(K_cache, V_cache, top_k, use_int8)` (4 args).
   - Fix `fused_kv_evict` signature/call (add optional `Q: Optional[Tensor] = None` or raise clear error / fall back cleanly when `Q is None`) and add a unit test in `tests/test_eviction.py`.
2. **Upgrade `benchmarks/bench_serving_regime.py`:**
   - Currently imports only `fused_gqa` (v3, 5.5% HBM peak), making `RESULTS.md` look 12x slower than SDPA.
   - Add `--kernel` CLI flag supporting `v3`, `v4`, `v4_fp8`, `triton_fp16`, `triton_fp8` (default `triton_fp8` if CUDA/Triton available, fallback `v4`), and ensure `save_result("serving_regime", ...)` records `"kernel"` in config/rows.
   - Update `benchmarks/make_report.py` so the serving-regime section prints which kernel produced the table.
3. **Add CI workflow `.github/workflows/ci.yml`:**
   - Python 3.9 & 3.11 matrix on `ubuntu-latest`.
   - Install `torch` (CPU), `triton`, `pytest`, `ruff`.
   - Run `ruff check sparse_kv triton_kernels benchmarks tests` (warn or fail on syntax/undefined names) and `TRITON_INTERPRET=1 pytest tests/test_triton_decode.py -v`.
4. **Quarantine non-inference clutter (do NOT delete data blindly; ask or move cleanly):**
   - Document in a short `CLEANUP_NOTES.md` or move `train/`, `jax_ref/`, `LLM_Inference_optimization.pdf` only if explicitly asked by user; otherwise update `README.md` repo layout so visitors don't confuse training scripts with the inference engine.

**Verification:**
- `python3 -m py_compile sparse_kv/*.py triton_kernels/*.py benchmarks/*.py tests/*.py`
- `TRITON_INTERPRET=1 pytest tests/test_triton_decode.py tests/test_eviction.py -v`

---

## WP-1: Paged KV Cache + Ragged Batching + Fused FP8 KV Append

**Goal:** Move from contiguous `[B, N, Hkv, D]` toy tensors to production PagedAttention KV layout with heterogeneous `seq_lens` and a fused KV quantize+scatter kernel.

**Files to read:**
- `triton_kernels/gqa_decode.py`
- `csrc/kernels/fp8_common.cuh`
- `csrc/kernels/kv_fp8.cu`
- `tests/test_triton_decode.py`

**Design Spec:**
- **Paged KV Pool Contract:**
  - `k_cache`, `v_cache`: `[num_blocks, block_size, Hkv, D]` (`torch.float16` or `torch.uint8` holding `float8_e4m3fn` bits). Set `block_size = 64` (aligned with `KV_PAGE_TOKENS = 64`, or support `block_size` in `{16, 64}`).
  - `k_scale`, `v_scale` (FP8 mode): `[num_blocks, block_size // KV_PAGE_TOKENS, Hkv]` (`torch.float32`; simplifies to `[num_blocks, Hkv]` when `block_size == KV_PAGE_TOKENS == 64`).
  - `block_table`: `[B, max_blocks_per_seq]` (`torch.int32`), mapping `(batch_idx, logical_block_idx) -> physical_block_idx`.
  - `seq_lens`: `[B]` (`torch.int32`), true token count per sequence in batch.
- **New Kernels:**
  1. `triton_kernels/paged_kv.py`:
     - `reshape_and_cache_fp8(k_new, v_new, k_cache, v_cache, k_scale, v_scale, slot_mapping)`: Triton kernel writing newly generated/prefilled `[num_tokens, Hkv, D]` FP16 K/V into paged FP8 cache slots and updating per-page absmax scales (or `kv_page_quant_append` for full blocks / incremental tail).
     - Also provide optional fused RoPE (`cos`, `sin`, `positions`) before quantize+store (`fused_rope_and_cache_fp8`), or clean helper `apply_rope_and_cache`.
  2. `triton_kernels/paged_gqa_decode.py`:
     - `paged_gqa_decode(q, k_cache, v_cache, block_table, seq_lens, scale=None, k_scale=None, v_scale=None, max_seq_len=None)`
     - Supports **variable `seq_lens`** across `B` (e.g., `seq_lens = [17, 64, 513, 4097]`), early-exiting splits past `seq_lens[b]`.
     - Supports `fp16` and `fp8` (`e4m3`) KV pools, arbitrary `G = Hq // Hkv` in `{1, 2, 4, 8}`, and `D` in `{64, 128}`.
     - Exact LSE combine kernel handling empty/masked splits cleanly.
  3. CUDA parity kernel (optional in WP-1 or follow-up): `csrc/kernels/paged_kv_append.cu` bound in `bindings.cpp`.

**Verification:**
- Create `tests/test_paged_gqa.py`:
  - Reference: gather logical sequences from shuffled `block_table` -> slice `[:seq_lens[b]]` -> `F.scaled_dot_product_attention(..., enable_gqa=True)`.
  - Test shuffled physical block IDs, heterogeneous `seq_lens = [1, 63, 64, 65, 257, 1025]`, `B in [1, 4]`, `D in [64, 128]`, `G in [1, 4, 8]`, both `fp16` and `fp8`.
  - Must pass under `TRITON_INTERPRET=1 pytest tests/test_paged_gqa.py -v` on CPU and on GPU.

---

## WP-2: `torch.library.custom_op` + CUDA Graphs + Real E2E Serving Benchmark vs FlashInfer / vLLM

**Goal:** Eliminate `torch.cuda._sleep` benchmark crutches and analytical projections (`step_ms*`); deliver real CUDA-Graph-captured end-to-end throughput/latency vs FlashInfer & vLLM.

**Files to read:**
- `sparse_kv/__init__.py`
- `triton_kernels/gqa_decode.py`
- `triton_kernels/paged_gqa_decode.py` (from WP-1)
- `benchmarks/bench_serving_regime.py`
- `modal_apps/bench_triton_h100.py`

**Tasks:**
1. **Register PyTorch Custom Ops (`sparse_kv/ops.py`):**
   - Wrap `fused_gqa_v4`, `fused_gqa_v4_fp8`, `triton_gqa_decode`, and `paged_gqa_decode` with `@torch.library.custom_op("sparse_kv::...", mutates_args=...)` + `@register_fake`.
   - Write `tests/test_cudagraph_compile.py` verifying:
     - `torch.cuda.CUDAGraph()` capture and replay over changing `seq_lens` / `block_table` contents with static buffer addresses.
     - `torch.compile(..., fullgraph=True)` produces zero graph breaks.
2. **Real End-to-End Model Decode Benchmark (`benchmarks/bench_e2e_serving.py`):**
   - Load `Qwen/Qwen3-8B` (ungated default; `--model meta-llama/Llama-3.1-8B` flag).
   - Run prefill -> capture CUDA Graph for single-step paged decode -> measure **real TPOT / ITL (ms/token) and total tok/s** across `B in [1, 4, 16, 32, 64]` and `ctx in [4096, 16384, 65536]`.
   - Compare: (a) PyTorch SDPA eager/compile, (b) `sparse_kv` FP16 paged + CUDA Graph, (c) `sparse_kv` FP8 paged + CUDA Graph.
3. **Head-to-Head vs FlashInfer & vLLM (`benchmarks/bench_vs_engines.py` + `modal_apps/bench_e2e_h100.py`):**
   - Optional dependency guards (`try: import flashinfer`, `try: import vllm`) so scripts still run cleanly when only one is installed.
   - Modal script installs `flashinfer-python` and `vllm`, runs matched H100 comparison at `B in [1, 4, 16, 32]`, `ctx in [4096, 16384, 65536]`, saves JSON via `bench_common.py`.

**Verification:**
- CPU/syntax: `python3 -m py_compile sparse_kv/ops.py benchmarks/bench_e2e_serving.py`
- GPU / Modal: `pytest tests/test_cudagraph_compile.py -v` and `modal run modal_apps/bench_e2e_h100.py`

---

## WP-3: CUDA Tensor-Core + `cp.async` Pipeline Decode Kernel (Close the CUDA vs Triton Gap)

**Goal:** Fix the Stage 4a instruction-issue bottleneck directly in CUDA (and/or CuTe) so hand-written CUDA matches or beats the Triton kernel and supports `S > 1` (speculative decode verification).

**Files to read:**
- `csrc/kernels/gqa_decode.cu` (lines 250–697 only)
- `csrc/kernels/fp8_common.cuh`
- `csrc/pybind/bindings.cpp`
- `setup.py`

**Tasks:**
1. **Implement `csrc/kernels/gqa_decode_tc.cu`:**
   - **Async global->shared pipeline:** Use `cp.async` (`asm volatile("cp.async.cg.shared.global ...")` + `cp.async.commit_group` / `cp.async.wait_group<1>`) with double-buffered shared memory tiles `[2][TILE_N][SMEM_STRIDE]` so tile `t+1` loads while tile `t` computes.
   - **Tensor-core QK^T and PV:** Pack `G` query heads (and optionally `S_q` query tokens for speculative decode verification, `G * S_q <= 16`) into the `M = 16` dimension of `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (or CuTe `TiledMMA` if `HAS_CUTLASS` is defined).
   - **Template parameters:** `template <int HEAD_DIM, int BLOCK_N, bool FP8>` supporting `HEAD_DIM in {64, 128}` and both FP16 and FP8 (`e4m3`) KV.
   - **Register/occupancy check:** Compile with `-Xptxas=-v` in an optional `PTXAS_VERBOSE=1` flag in `setup.py` to log zero local-memory spills.
2. **Bind in `csrc/pybind/bindings.cpp`:**
   - Expose `fused_gqa_tc` and `fused_gqa_tc_fp8` (supporting both `Q [B, Hq, D]` and `Q [B, S_q, Hq, D]` for speculative decode).
3. **Add `tests/test_gqa_tc.py` and extend `benchmarks/bench_triton_decode.py`:**
   - Assert `max_abs_err < 2e-3` vs `fp32` SDPA for `N in [1, 63, 64, 65, 1000, 16384]`, `B in [1, 4]`, `S_q in [1, 4]`.

**Verification:**
- `PTXAS_VERBOSE=1 python3 setup.py build_ext --inplace` (0 register spills)
- `pytest tests/test_gqa_tc.py -v`

---

## WP-4: Real Quest-Style Page-Sparse Decode (Finish WS4 — True HBM Reduction)

**Goal:** Replace the dead `sparse_attention.cu` ablation (which reads all of K before selecting top-k) with genuine two-stage page-sparse paged attention that cuts HBM traffic proportional to `top_k_pages / total_pages`.

**Files to read:**
- `docs/PLAN.md` (WS4 section only)
- `triton_kernels/paged_gqa_decode.py` (from WP-1)
- `benchmarks/quality_gate_fp8.py`

**Tasks:**
1. **Per-page metadata:** Maintain `k_min`, `k_max` `[num_blocks, Hkv, D]` (`fp16`) updated during `reshape_and_cache_*`.
2. **Page-scorer + sparse paged decode (`triton_kernels/quest_sparse_decode.py`):**
   - Stage 1 kernel: Given `q [B, Hq, D]`, compute upper-bound page score per logical page `p`:
     `score[b, h_kv, p] = max_{g in group} sum_d max(q_{g,d} * k_min_{p,d}, q_{g,d} * k_max_{p,d})`.
   - Select `top_k_pages` indices (always retaining page `0` for attention sinks and the last `recent_pages` pages for local window).
   - Stage 2 kernel: Pass `selected_pages [B, Hkv, top_k_pages]` into `paged_gqa_decode` so **only** those pages are loaded from HBM.
3. **Quality + Speedup Gate (`benchmarks/quality_gate_quest.py`):**
   - Measure **effective HBM bytes read**, kernel latency vs dense Triton at `64K` (`budget = 12.5%, 25%`), and **teacher-forced perplexity / confident-step parity + Needle-in-a-Haystack retrieval** on `Qwen/Qwen3-8B` (keeping first `N` dense layers configurable, e.g. `dense_prefix_layers=2`).

**Verification:**
- `TRITON_INTERPRET=1 pytest tests/test_quest_sparse.py -v`
- On H100: verify latency scales linearly with `top_k_pages` and perplexity delta stays within budget.

---

## WP-5: H100 Nsight Compute (`ncu`) + Nsight Systems (`nsys`) Roofline & Multi-GPU TP Profiling

**Goal:** Replace RTX 4060 Laptop `ncu.txt` and training `nsys` traces with genuine **H100 inference profiling artifacts** and roofline analysis.

**Files to read:**
- `benchmarks/profile_ncu.py`
- `profiles/README.md`
- `results/triton/nvidia-geforce-rtx-4060-laptop-gpu/ncu.txt`

**Tasks:**
1. **Expand `benchmarks/profile_ncu.py`:**
   - Add `--impl` options: `v4`, `v4_fp8`, `tc`, `tc_fp8`, `triton`, `triton_fp8`, `paged_triton`, `flashinfer`.
   - Collect Roofline + Stall metrics:
     - `gpu__time_duration.sum`
     - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
     - `dram__bytes_read.sum`
     - `sm__inst_issued.avg.pct_of_peak_sustained_active`
     - `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`
     - `smsp__warp_issue_stalled_long_scoreboard_per_instruction_executed.avg`
     - `smsp__warp_issue_stalled_mio_throttle_per_instruction_executed.avg`
     - `lts__t_sector_hit_rate.pct`
   - Add `--sass` mode running `cuobjdump -sass` on `sparse_kv/_C*.so` and summarizing `HMMA`/`LDGSTS`/`LDG.E.128` instruction counts.
2. **Full-step `nsys` profiling script (`benchmarks/profile_e2e_nsys.py`):**
   - Annotate `Qwen3-8B` decode step with NVTX ranges (`RMSNorm`, `QKV_Proj`, `RoPE_KV_Append`, `Paged_Attention`, `O_Proj`, `MLP_SwiGLU`, `AllReduce` if `TP > 1`).
   - Output clean Markdown breakdown table in `profiles/INFERENCE_PROFILE.md` showing exact % of decode step spent in each component across `B = 1, 16, 64` at `ctx = 4K, 64K`.
3. **Optional Multi-GPU TP=2/4 Overlap Benchmark (`benchmarks/bench_tp_decode.py`):**
   - `torch.distributed` NCCL `AllReduce` + decode layer timing with `nsys` showing compute vs comm ratio and CUDA Graph capture over NCCL (`torch.cuda.CUDAGraph` with NCCL >= 2.19).
