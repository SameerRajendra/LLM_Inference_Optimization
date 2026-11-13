# Profiling Traces

This directory stores the **H100 Inference Profiling Runbook (WP-5)** alongside a short note on an earlier distributed training trace.
Actual captured H100 SASS and Nsight Systems (`nsys`) artifacts live in [`../results/profiles/`](../results/profiles/):
- `results/profiles/h100_sass.md` — `cuobjdump -sass` instruction summary (`HMMA`, `LDGSTS`, `LDG.E.128`) for `sm_90a`
- `results/profiles/h100_inference_profile.md` — CUDA-events NVTX stage attribution (`Qwen3-8B`, `paged_fp8`)
- `results/profiles/h100_nsys_profile.md` — `nsys stats --report nvtx_sum` E2E decode stage breakdown
- `results/profiles/h100_nsys_kern_sum.txt` — `nsys stats --report cuda_gpu_kern_sum` per-kernel GPU timings
- `results/profiles/h100_tp2_nsys_stats.txt` — `nsys` `TP=2` NCCL + paged kernel stats

*(Note: `ncu` hardware PMU capture is skipped on Modal because container permissions block `ERR_NVGPUCTRPERM`; [`stage1_64k_ncu.txt`](stage1_64k_ncu.txt) is retained as a historical Stage 1 log.)*

---

## H100 Inference Profiling Runbook (WP-5)

### 1. Nsight Compute (`ncu`) Roofline, Stalls & SASS Inspection

`benchmarks/profile_ncu.py` brackets a single decode kernel launch with
`cudaProfilerStart`/`cudaProfilerStop` and collects roofline, tensor-pipe,
L2 cache, occupancy, and warp-stall counters across implementations (`v3`, `v4`,
`v4_fp8`, `tc`, `tc_fp8`, `triton`, `triton_fp8`, `paged_triton`, `paged_fp8`,
`quest`, `quest_fp8`, `flashinfer`, `sdpa`).

```bash
# Default ncu roofline + stall sweep (requires host GPU counter permissions)
python benchmarks/profile_ncu.py --ncu --ctx 32768

# Parse `ncu --csv` output into a Markdown summary table and save raw + formatted output
python benchmarks/profile_ncu.py --ncu --ncu-csv --ctx 32768 --ncu-out results/profiles/NCU_PROFILE.md

# Profile a specific kernel implementation with custom ncu sections
ncu --profile-from-start off -k regex:_decode_split_kernel -c 1 --section SpeedOfLight \
    python benchmarks/profile_ncu.py --impl triton_fp8 --ctx 32768

# Inspect compiled CUDA SASS instructions (HMMA / LDGSTS / LDG.E.128) via cuobjdump
python benchmarks/profile_ncu.py --sass --sass-out results/profiles/h100_sass.md
```

Collected `ncu` metrics:
- `gpu__time_duration.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `dram__bytes_read.sum`
- `sm__inst_issued.avg.pct_of_peak_sustained_active`
- `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`
- `smsp__warp_issue_stalled_long_scoreboard_per_instruction_executed.avg`
- `smsp__warp_issue_stalled_mio_throttle_per_instruction_executed.avg`
- `lts__t_sector_hit_rate.pct`
- `sm__warps_active.avg.pct_of_peak_sustained_active`

### 2. Nsight Systems (`nsys`) End-to-End Decode Breakdown

`benchmarks/profile_e2e_nsys.py` annotates a `Qwen3-8B` single-step decode
pass with NVTX ranges (`Decode_Step` outer range containing `Embedding`,
`RMSNorm`, `QKV_Proj`, `RoPE_KV_Append`, `Quest_Page_Select`,
`Paged_Attention`, `O_Proj`, `MLP_SwiGLU`, and `LM_Head`) and writes a
per-component latency/percentage breakdown table.

```bash
# Run NVTX-annotated nsys sweep across batch sizes and context lengths and write Markdown report
python benchmarks/profile_e2e_nsys.py --nsys-sweep --out results/profiles/h100_nsys_profile.md

# Capture a standalone Nsight Systems trace for a single decode configuration
nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
    --capture-range-end=stop -o /tmp/e2e_decode_b16_64k \
    python benchmarks/profile_e2e_nsys.py --method paged_fp16 --batch 16 --ctx 65536

# Lightweight CUDA-events fallback sweep when nsys is unavailable
python benchmarks/profile_e2e_nsys.py --nsys-sweep --cuda-events --out results/profiles/h100_inference_profile.md
```

### 3. Multi-GPU Tensor-Parallel (`TP >= 2`) Decode & NCCL Overlap

`benchmarks/bench_tp_decode.py` measures `torch.distributed` NCCL `AllReduce`
(`TP_AllReduce_Attn`, `TP_AllReduce_MLP`) and decoder-layer compute overlap
(`TP_Paged_Attn`, `TP_MLP_SwiGLU`), including CUDA Graph capture over NCCL
(when NCCL >= 2.19).

```bash
# Run TP=2 decode benchmark via standalone torch.multiprocessing.spawn
python benchmarks/bench_tp_decode.py --tp 2

# Launch via torchrun (e.g., TP=4)
torchrun --nproc_per_node=4 benchmarks/bench_tp_decode.py --shape qwen3-8b

# Capture an nsys trace bracketed around steady-state TP=2 decode steps
nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
    -o /tmp/tp2_decode \
    torchrun --nproc_per_node=2 benchmarks/bench_tp_decode.py --nsys-bracket --steps 16

# CPU-only partition and shard-equivalence validation (no GPUs required)
python benchmarks/bench_tp_decode.py --dry-run-cpu --shape qwen3-8b --tp 2
```

---

## Legacy Training Trace Note (`archive/train/train_fsdp_lora.py`)

Summary retained from an earlier distributed training capture on dual NVIDIA H200 NVLink
(unrelated to the inference engine benchmarks above; raw `.nsys-rep` binary removed from git):

| Metric | Value |
|--------|-------|
| Model | 4× `LlamaStyleDecoderLayer` (dim=8192, SwiGLU MLP), LoRA rank=16 |
| AllGather avg / ReduceScatter avg | 22.0 ms / 22.9 ms |
| NCCL % of wall time | 3.2% (`~30:1` compute-to-comm ratio vs 45–162 ms GEMMs) |
