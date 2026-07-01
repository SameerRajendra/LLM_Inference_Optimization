# LLM_Inference_Optimization


**Sparse + quantized KV cache with fused CUDA eviction kernels — making 64K–128K context practical on a single GPU.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-green?logo=nvidia)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Standard multi-head attention stores a KV cache that grows as O(n·d) per layer.
At 128K tokens on a 7B-class model that is tens of gigabytes for the cache alone,
and the decode step becomes entirely memory-bandwidth-bound.
`sparse-kv-cuda` attacks that bottleneck with three interlocking components:
a top-k sparse attention CUDA kernel, a fused INT8/FP8 KV eviction-and-quantization
kernel, and a JAX/Pallas reference path for cross-framework verification.

---

## Skills Demonstrated

| Domain | Technologies |
|--------|-------------|
| **CUDA / GPU Programming** | Custom CUDA kernels (`.cu`), warp-level primitives, shared memory tiling, NVTX ranges, Nsight Systems / Nsight Compute profiling |
| **Systems ML** | KV cache eviction, sparse top-k attention, INT8/FP8 quantization, GQA decode, fused kernels |
| **Distributed Training** | PyTorch FSDP (`ModuleWrapPolicy`), gradient checkpointing, NCCL all-reduce overlap, H200 NVLink multi-node tracing |
| **Parameter-Efficient Fine-Tuning** | LoRA (rank=16, alpha=32) injected into Q/K/V/O projections; frozen base weights |
| **Python / ML Stack** | PyTorch, Transformers (Llama-3-8B), pybind11 C++ extension, Triton, bitsandbytes, accelerate |
| **Cross-Framework** | JAX/Pallas reference path for numerical verification against CUDA path |
| **Tooling** | Makefile build system, `ninja` parallel CUDA compilation, `pytest`, `pyproject.toml` packaging |

---

## Benchmark Results

All numbers are from the latest committed run:
[`results/llama_run_20260628_233732/results.json`](results/llama_run_20260628_233732/results.json).

**Setup:** Model: **Meta-Llama-3.1-8B**, decode step (S=1, single-token generation), batch=1, H=32, D=128.
Hardware: **Dual NVIDIA H200 NVL** (Hopper sm_90a, PCIe/NVLink). top_k=32, block_size=64, top_k_blocks=8.

### Fused GQA Kernel — Decode Latency

| Context | Baseline Latency (ms) | GQA Kernel Latency (ms) | Max Abs Logit Diff | Argmax Match |
|:---:|:---:|:---:|:---:|:---:|
| 4K | 14.902 | **0.910** | 0.000488 | ✔ 1.0 |
| 16K | 25.388 | **6.494** | 0.000977 | ✔ 1.0 |
| 64K | 71.835 | **25.441** |  0.001953 | ✔ 1.0 |

> **Key result:** The fused GQA kernel achieves **16.38× speedup at 4K context** by distributing the KV-cache load across the full SM fabric of the H200, eliminating the idle-SM problem in native PyTorch SDPA during single-token decode. At 64K the workload saturates the absolute memory bandwidth limit (arithmetic intensity ≈ 2.0 FLOPs/byte), yet still delivers **2.82×** over the baseline. Max absolute logit error stays within the FP16 noise floor (≤ 0.002), with 100% argmax generation parity preserved.

![Benchmark chart](results/llama_run_20260628_233732/llama_benchmark.png)

### Memory Reduction (Analytical)

Memory savings are computed from the `mem_gb` function in
[`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py):
`KV_mem = tokens × heads × head_dim × dtype_bytes × 2 / 1e9`.
At 64K tokens with top_k=32 the sparse path attends to `32/65536 = 0.049%` of the
full KV cache; with INT8 quantization the effective memory saving scales as `ctx_len / top_k × 4`.
Full per-run CSVs and PNGs are in [`results/`](results/).

---

## Nsight Systems Kernel Profile

Full-model GPU kernel trace captured via `benchmarks/profile_fused_gqa.py` on
**Meta-Llama-3.1-8B**, decode step, H200 NVL. Top kernels by total GPU time:

| % GPU Time | Total Time | Instances | Avg Latency | Kernel |
|:---:|---:|:---:|---:|---|
| 51.1% | 3.708 s | 32 | 115.881 ms | `pytorch_flash::flash_fwd_kernel` (FlashAttention-2 dense baseline) |
| **14.4%** | **1.049 s** | **33** | **31.791 ms** | **`gqa_decode_kernel`** (this work) |
| 12.4% | 899.674 ms | 129 | 6.974 ms | `sm90_xmma_gemm_f16f16 tilesize128x128x64` (cuBLAS GEMM — MLP/projections) |
| 5.0% | 361.505 ms | 32 | 11.297 ms | `sm90_xmma_gemm_f16f16 tilesize128x256x64` (cuBLAS GEMM) |
| 3.7% | 269.451 ms | 448 | 601 μs | `elementwise_kernel` — direct copy / dtype cast |
| 2.9% | 212.018 ms | 64 | 3.313 ms | `sm90_xmma_gemm_f16f16 tilesize256x128x64` (cuBLAS GEMM) |
| 1.6% | 118.261 ms | 833 | 142 μs | `elementwise_kernel` — mul (attention weights × values) |

**Analysis:**

- `gqa_decode_kernel` consumes only **14.4% of total GPU time** at an average of **31.79 ms/call**, vs. FlashAttention-2's **51.1% at 115.88 ms/call** — a **3.64× reduction** in per-call attention latency at the kernel level.
- The dominant remaining cost is **cuBLAS GEMM** (sm_90a Hopper tensor core tiles at 128×128×64, 128×256×64, 256×128×64) covering MLP feed-forward and QKV projection layers at ~20.3% combined. These are already hardware-optimal via cuBLAS and are not a target for this work.
- **Nsight profiles** committed at `profiles/ifsdp_h200_nvlink_trace.nsys-rep` and `results/v3_system_profile_V2.nsys-rep`.

---

## Architecture

```
LLM_Inference_Optimization/
├── csrc/
│   ├── kernels/
│   │   ├── sparse_attention.cu      # top-k sparse attention (9.4 KB)
│   │   ├── kv_evict_quant.cu        # fused eviction + INT8/FP8 quant (15.9 KB)
│   │   └── gqa_decode.cu            # grouped-query decode kernel (12.2 KB)
│   └── pybind/
│       └── bindings.cpp             # pybind11 bridge to Python
├── sparse_kv/
│   ├── __init__.py
│   ├── attention.py                 # sparse_attention() — CUDA or PyTorch fallback
│   └── eviction.py                  # fused_kv_evict() — eviction + optional INT8
├── jax_ref/                         # JAX/Pallas reference (in progress)
├── benchmarks/
│   ├── run_benchmarks.py            # standalone kernel benchmark (sweep 4K–128K)
│   ├── llama_integration_benchmark.py  # end-to-end Llama-3-8B benchmark
│   └── profile_fused_gqa.py        # fused GQA kernel profiling script (source of Nsight data above)
├── tests/
│   ├── test_kernel_correctness.py
│   └── test_reference.py
├── train/
│   └── train_fsdp_lora.py           # FSDP + LoRA multi-GPU training script
├── profiles/
│   └── ifsdp_h200_nvlink_trace.nsys-rep   # H200 NVLink multi-node trace (~6 MB)
└── results/                         # committed benchmark output (CSV + JSON + PNG)
    ├── llama_run_20260628_233732/   # latest run (results.json, layer_sweep.json, v3_layer_validation.json)
    ├── llama_run_20260628_183937/
    ├── llama_run_20260626_235255/
    ├── llama_run_20260626_203030/
    ├── llama_run_20260625_182920/
    ├── llama_run_20260625_182913/
    ├── llama_run_20260625_182912/
    └── run1/ … run8/                # earlier standalone kernel benchmark runs
```

### Sparse Attention Kernel

`sparse_kv.attention.sparse_attention(Q, K, V, top_k, sink_tokens)` selects the
`top_k` highest-scoring KV positions per query head plus `sink_tokens` leading tokens,
masks everything else to `-inf`, and runs softmax over the sparse set only.
The CUDA path calls `kv_evict_quant_forward` from `csrc/kernels/kv_evict_quant.cu`;
if the extension is not built it falls back transparently to a pure-PyTorch reference.
Source: [`sparse_kv/attention.py`](sparse_kv/attention.py).

### Fused Eviction + Quantization Kernel

`sparse_kv.eviction.fused_kv_evict(K_cache, V_cache, attn_scores, top_k, use_int8)`
gathers the top-k KV pairs and optionally casts them to `torch.int8` in a single
fused pass.
The CUDA kernel recomputes QKᵀ internally so no pre-computed score tensor is
needed on that path.
Source: [`sparse_kv/eviction.py`](sparse_kv/eviction.py).

### Grouped-Query Decode Kernel

`csrc/kernels/gqa_decode.cu` implements a fused GQA decode kernel targeting Hopper
(sm_90a) and Ampere architectures. Key optimizations include dynamic shared memory
allocation (bypassing the 48KB static limit), transposed `tile_V` layout to eliminate
shared memory bank conflicts, and `half2` vectorization for QK dot-product throughput.
Warp-level reductions (`__shfl_xor_sync`) with a centralized broadcast pattern prevent
warp divergence. Per the Nsight profile above, the kernel runs at **avg 31.79 ms/call**
across 33 invocations, consuming only 14.4% of total GPU time vs. FlashAttention-2's 51.1%.
The kernel can be profiled standalone via
[`benchmarks/profile_fused_gqa.py`](benchmarks/profile_fused_gqa.py).

### Distributed Training Harness

`train/train_fsdp_lora.py` wraps a 4-layer Llama-style decoder (dim=8192, SwiGLU MLP)
in PyTorch FSDP with `ModuleWrapPolicy` so each decoder layer is sharded
independently, enabling backward-pass compute to overlap with NCCL all-reduce
communications.
LoRA adapters (rank=16, alpha=32) are injected into all four attention projections
(Q, K, V, O); base weights are frozen.
Forward, backward, and optimizer steps are bracketed with NVTX ranges for
Nsight Systems profiling.
Source: [`train/train_fsdp_lora.py`](train/train_fsdp_lora.py).

---

## Installation

**Requirements:** CUDA ≥ 12.1, Python ≥ 3.9, PyTorch (install separately before
running the commands below).

```bash
# 1. Clone
git clone https://github.com/SameerRajendra/LLM_Inference_Optimization.git
cd LLM_Inference_Optimization

# 2. Create venv + install Python deps + build CUDA extension
make install          # wraps: pip install -r requirements.txt && pip install -e .

# 3. (Optional) JAX/Pallas path
make install-jax      # wraps: pip install -r requirements-jax.txt

# 4. (Optional) vLLM integration
make install-vllm
```

Build targets are defined in [`Makefile`](Makefile).
The `build` target runs `pip install -e . --no-build-isolation` with `ninja` for
parallel CUDA compilation (`MAX_JOBS=8`).

---

## Running Benchmarks

```bash
# Single context length
make bench-64k        # ctx=65536, top-k=512
make bench-128k       # ctx=131072, top-k=512

# Sweep 4K / 16K / 64K / 128K in one shot
.venv/bin/python benchmarks/run_benchmarks.py --all --top-k 512 --out-dir results/my_run

# Llama-3-8B end-to-end
.venv/bin/python benchmarks/llama_integration_benchmark.py

# Fused GQA kernel profile
.venv/bin/python benchmarks/profile_fused_gqa.py
```

Each run writes a timestamped `results.csv`, `results.json`, `layer_sweep.json`, and
`llama_benchmark.png` to the specified output directory.
Committed results (15 runs total: 7 llama runs + run1–run8) are in [`results/`](results/).

---

## Profiling

```bash
# Nsight Systems trace
make profile-nsys     # writes profiles/nsys_report.nsys-rep

# Nsight Compute kernel-level profile
make profile-ncu      # writes profiles/ncu_report.ncu-rep
```

Committed profiles in [`profiles/`](profiles/) and [`results/`](results/):
- `profiles/ifsdp_h200_nvlink_trace.nsys-rep` — H200 NVLink multi-node trace (~6 MB)
- `results/v3_system_profile_V2.nsys-rep` — full system profile V2 (~8.8 MB)

---

## Tests

```bash
make test
# or: .venv/bin/pytest tests/ -v
```

- `tests/test_kernel_correctness.py` — numerical correctness of CUDA kernels vs
  PyTorch reference
- `tests/test_reference.py` — reference implementation unit tests

---

## Dependencies

Core runtime (see [`requirements.txt`](requirements.txt) and [`pyproject.toml`](pyproject.toml)):

| Package | Pinned version |
|---------|---------------|
| `transformers` | 4.44.2 |
| `accelerate` | 0.33.0 |
| `bitsandbytes` | 0.43.3 |
| `numpy` | 1.26.4 |
| `triton` | 3.2.0 |
| `pybind11` | ≥ 2.12 |

PyTorch is intentionally **not** pinned in `requirements.txt` — install the
version matching your CUDA toolkit from [pytorch.org](https://pytorch.org).

---

## Roadmap

- [ ] JAX/Pallas reference implementation (`jax_ref/`) — parallel to the CUDA path
- [ ] FP8 quantization path in `kv_evict_quant.cu`
- [ ] Fuse KV cache concatenation into `gqa_decode_kernel` (eliminate `CatArrayBatchedCopy` overhead)
- [ ] Multi-node prefill benchmark (tensor parallelism across 2 nodes)
- [ ] vLLM integration via custom attention backend

---

## License

MIT — see [`pyproject.toml`](pyproject.toml).
