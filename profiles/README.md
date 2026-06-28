# Profiling Traces

This directory contains Nsight Systems (`.nsys-rep`) traces captured during
distributed training of the FSDP + LoRA harness on dual NVIDIA H200 NVLink.
The table below extracts the key communication metrics so the results are
readable without opening the Nsight GUI.

---

## Hardware

| Field | Value |
|-------|-------|
| Node | g213 |
| GPUs | Dual NVIDIA H200 NVL |
| Interconnect | NVLink |
| OS | Rocky Linux 9.3 |
| Driver | 580.82.07 |

---

## NCCL Communication Summary

| Trace | Profiled Steps | AllGather avg | ReduceScatter avg | NCCL % of wall time |
|-------|:--------------:|:-------------:|:-----------------:|:-------------------:|
| [`fsdp_baseline.nsys-rep`](fsdp_baseline.nsys-rep) | 1 | 4.9 ms | 13.4 ms | 46.0% |
| [`ifsdp_h200_nvlink_trace.nsys-rep`](ifsdp_h200_nvlink_trace.nsys-rep) | ~12 | 22.0 ms | 22.9 ms | 3.2% |

### Reading the numbers

- **AllGather** — FSDP reconstructs sharded weight parameters before each
  forward pass via AllGather; lower is better and reflects NVLink bandwidth.
- **ReduceScatter** — gradients are scattered and reduced across ranks after
  backward; dominates the baseline trace because compute and comm are not
  overlapped.
- **NCCL % of wall time** — fraction of total step time spent blocked in NCCL
  collectives.
  - `fsdp_baseline.nsys-rep` (46.0%) shows a single un-optimized step where
    communication is the dominant bottleneck.
  - `ifsdp_h200_nvlink_trace.nsys-rep` (3.2%) shows ~12 steps with FSDP
    `ModuleWrapPolicy` active: each decoder layer is sharded independently,
    so backward-pass compute overlaps with NCCL communications for adjacent
    layers, collapsing the communication fraction from 46% → 3.2%.

---

## How to Open

```bash
# Nsight Systems CLI (no GUI required)
nsys stats profiles/fsdp_baseline.nsys-rep
nsys stats profiles/ifsdp_h200_nvlink_trace.nsys-rep

# Or open interactively
nsys-ui profiles/ifsdp_h200_nvlink_trace.nsys-rep
```

The training script that generated these traces is
[`train/train_fsdp_lora.py`](../train/train_fsdp_lora.py).
NVTX ranges (`Forward_Pass_Compute`, `Backward_Pass_Compute_and_Comm_Overlap`,
`Optimizer_Parameter_Updates`) are annotated in the trace for precise
per-phase attribution.
