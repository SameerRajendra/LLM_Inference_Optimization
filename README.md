# LLM_Inference_Optimization

**A split-KV (FlashDecoding-style) fused GQA decode kernel for Llama-3.1-8B on NVIDIA Hopper — benchmarked   against FlashAttention, and the foundation for an adaptive KV-cache decode engine.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green?logo=nvidia)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

Long-context LLM decoding is bottlenecked by KV-cache memory bandwidth: at 64K tokens the
decode step is memory-bound, not compute-bound. This repo builds and **rigorously benchmarks**
custom CUDA decode kernels for Llama-3.1-8B (32 query / 8 KV heads, `head_dim=128`, Hopper
`sm_90a`), measured against a state-of-the-art baseline (PyTorch SDPA / FlashAttention).

> **Status.** The dense **split-KV GQA decode kernel is complete and validated.** The sparse
> and quantized KV kernels and the adaptive per-layer router are **in progress** — see
> [Roadmap](#roadmap). This README reports only measured results.

---

## Key results

Hardware: **NVIDIA H100 SXM** (Hopper `sm_90a`). Model: **Meta-Llama-3.1-8B**, single-token
decode (S=1), batch=1.

### Split-KV GQA decode kernel

Partitions the KV sequence across all SMs (FlashDecoding-style) and merges partial softmax
states with an exact log-sum-exp combine kernel — fixing the single-block design that left
most of the GPU idle.

- **Numerically exact** vs PyTorch SDPA across **all 32 layers**: max abs error ≤ **1e-3** up to 64K context.
- **18× faster than a naïve single-block decode kernel** (isolated kernel latency at 64K: 25.4 ms → **1.37 ms**).

### Comparison vs FlashAttention (isolated kernel, per decode step)

| Context | split-KV kernel (ms) | PyTorch SDPA / FlashAttention (ms) | our HBM bandwidth |
|:---:|:---:|:---:|:---:|
| 4K  | 0.075 | 0.019 | ~6% of peak |
| 16K | 0.402 | 0.038 | ~5% of peak |
| 64K | 1.359 | 0.101 | ~6% of peak |

FlashAttention reaches **~77% of H100 HBM bandwidth** — it sits near the memory roofline for
exact attention, thanks to an asynchronous memory pipeline (`cp.async`/TMA + warp
specialization). **This project does not claim to beat FlashAttention.** The comparison above
is used to characterize the decode roofline and to locate where real gains are available.
Reproduce with [`benchmarks/profile_kernel_vs_sdpa.py`](benchmarks/profile_kernel_vs_sdpa.py).

### Where decode time actually goes

End-to-end profiling shows that at batch=1, **attention is only ~14% of decode latency** —
weight-GEMMs (reading the 16 GB of model weights) and kernel-launch overhead dominate. The
implication drives the roadmap: long-context decode gains come from **reducing bytes moved**
(KV-cache quantization, query-aware sparsity), not from a faster exact-attention kernel.

### Layer-sensitivity sweep

Applying top-k attention sparsity to the first *N* layers and measuring logit drift vs dense:
argmax parity holds through **16 sparse layers and collapses at 32**. This echoes the
*reconstruction-error-explosion* concept (Huang et al., ICML 2025, arXiv:2502.14770 — from the
weight-pruning setting) and motivates a **per-layer dense/sparse routing policy**.

---


The sparse kernels are included as **ablations**: because they compute scores over the
full KV before selecting top-k, they do not yet reduce bandwidth and are slower than dense. The
Quest-style redesign (page-level criticality estimation → load only top-k pages) is what turns
sparsity into a real speedup.

---

## Skills demonstrated

| Domain | Details |
|---|---|
| **CUDA / Hopper** | Split-KV / FlashDecoding-style kernel, two-pass online-softmax combine, warp + block reductions (`__shfl_xor_sync`), dynamic shared memory (>48 KB via `cudaFuncSetAttribute`), `half2` vectorization |
| **Performance analysis** | Nsight Systems, roofline / effective-bandwidth analysis, isolated-vs-end-to-end benchmarking, bottleneck diagnosis against a SOTA baseline |
| **LLM inference internals** | GQA decode, KV cache, online softmax, long-context memory-bandwidth analysis |
| **Python / ML stack** | PyTorch, HuggingFace Transformers (Llama-3.1-8B), pybind11 C++ extension, `ninja` CUDA builds |
| **Cross-framework** | JAX/Pallas reference kernel for numerical verification |

---

## Architecture

```
LLM_Inference_Optimization/
├── csrc/
│   ├── kernels/
│   │   ├── gqa_decode.cu          # split-KV GQA decode + combine kernels (this work)
│   │   ├── sparse_attention.cu    # block-sparse decode (ablation)
│   │   └── kv_evict_quant.cu      # top-k sparse / eviction (ablation)
│   └── pybind/bindings.cpp        # pybind11 bridge
├── sparse_kv/                     # Python package (sparse_kv._C extension)
├── jax_ref/                       # JAX/Pallas reference kernel
├── benchmarks/
│   ├── llama_integration_benchmark.py  # end-to-end Llama-3.1-8B decode benchmark
│   ├── profile_kernel_vs_sdpa.py       # isolated kernel vs FlashAttention (bandwidth)
│   └── validate_int8_kv.py             # INT8 KV-cache accuracy gate
├── tests/                         # kernel correctness vs PyTorch reference
├── train/
│   └── train_fsdp_lora.py             # FSDP + custom LoRA fine-tuning harness
└── results/                       # committed benchmark output (CSV + JSON)
```

### Split-KV GQA decode kernel

[`csrc/kernels/gqa_decode.cu`](csrc/kernels/gqa_decode.cu) implements two passes:
`gqa_decode_splitkv_kernel` (each block attends one KV chunk of one head, emitting an
unnormalized partial + online-softmax state) and `gqa_combine_kernel` (exact log-sum-exp
merge across chunks). `num_splits` is chosen from the SM count and a per-split token target so
the whole GPU is used even at batch=1. Exposed as `sparse_kv._C.fused_gqa(Q, K, V, scale)`.

---

## Installation

**Requirements:** CUDA 12.4 toolkit (`nvcc` on `PATH`), Python ≥ 3.9, an NVIDIA GPU. Install
**PyTorch first** (it is intentionally not in `requirements.txt`), then build with
`--no-build-isolation` (the build imports torch).

```bash
git clone https://github.com/SameerRajendra/LLM_Inference_Optimization.git
cd LLM_Inference_Optimization

python -m pip install --upgrade pip setuptools wheel ninja pybind11
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e . --no-build-isolation -v        # builds the CUDA extension

# optional: JAX/Pallas reference path
pip install -r requirements-jax.txt
```

---

## Running the benchmarks

```bash
# isolated kernel vs FlashAttention (no model download — synthesizes tensors)
python benchmarks/profile_kernel_vs_sdpa.py

# end-to-end Llama-3.1-8B decode (needs HF access to meta-llama/Llama-3.1-8B)
python benchmarks/llama_integration_benchmark.py

# INT8 KV-cache accuracy gate
python benchmarks/validate_int8_kv.py
```

Each end-to-end run writes a timestamped `results.csv` / `results.json` under `results/`.

---

## Tests

```bash
pytest tests/ -v
```

---

## Roadmap — toward an adaptive KV-cache decode engine

- [ ] **Quest-style query-aware sparse kernel** — page-level criticality → load only top-k pages (real byte reduction)
- [ ] **INT8/FP8 KV-cache quantization kernel** — KIVI-style scales; CUTLASS FP8 GEMM on Hopper
- [ ] **Adaptive per-layer router** — dense / sparse / quantized per layer, grounded in the reconstruction-error analysis above
- [ ] **Autotuner** — multi-objective (NSGA-II) search over split size, page budget, and quantization bits
- [ ] **Tensor-parallel (TP=2) decode** — head + KV-cache sharding with NCCL all-reduce over NVLink

---

## References

1. Dao et al., *FlashAttention*, NeurIPS 2022. · FlashDecoding (Dao et al., 2023).
2. Ainslie et al., *GQA*, EMNLP 2023.
3. Tang et al., *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*, ICML 2024.
4. Huang et al., *Determining Layer-wise Sparsity for LLMs Through a Theoretical Perspective*, ICML 2025 (arXiv:2502.14770).
5. Liu et al., *KVTuner*, arXiv:2502.04420. · Yao et al., *TailorKV*, Findings of ACL 2025.

## License

MIT — see [`pyproject.toml`](pyproject.toml).
