"""Minimal FSDP profiling script — throwaway baseline.

Purpose: capture a single FSDP optimization step on a small synthetic
workload to verify Nsight Systems tracing is working and establish a
non-overlapped communication baseline.

This is NOT the same experiment as train_fsdp_lora.py.
Key differences vs train_fsdp_lora.py:

| Property         | fsdp_profile.py          | train_fsdp_lora.py               |
|------------------|--------------------------|----------------------------------|
| Model            | 10x nn.Linear(8192,8192) | 4x LlamaStyleDecoderLayer + LoRA |
| Input shape      | [32, 8192]               | [8, 2048, 8192]  (512x larger)   |
| FSDP wrap        | Default (whole model)    | ModuleWrapPolicy per layer       |
| LoRA adapters    | None                     | Yes (rank=16, alpha=32)          |
| Warmup           | None                     | 2 iterations                     |
| Steps profiled   | 1                        | ~12                              |
| NVTX range       | FSDP_Step                | End_to_End_FSDP_Optimization_Step|

The default FSDP wrapping here shards the whole model as one unit —
it does NOT enable layer-boundary compute-communication overlap.
That mechanism is only active in train_fsdp_lora.py with ModuleWrapPolicy.

Generated trace: profiles/fsdp_baseline.nsys-rep
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.cuda.nvtx import range_push, range_pop


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def main():
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Minimal 10-layer MLP — no LoRA, no decoder structure
    model = nn.Sequential(
        *[nn.Linear(8192, 8192, bias=False) for _ in range(10)]
    ).to(device)

    # Default FSDP: whole model as a single shard — no layer-boundary overlap
    sharded_model = FSDP(model, device_id=local_rank)

    optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Small synthetic input — [32, 8192], 262,144 elements (512x smaller than full run)
    inputs  = torch.randn(32, 8192, device=device)
    targets = torch.randn(32, 8192, device=device)

    # No warmup — profiling starts immediately
    torch.cuda.synchronize()
    dist.barrier()

    range_push("FSDP_Step")

    outputs = sharded_model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    range_pop()  # End FSDP_Step

    if local_rank == 0:
        print("[+] fsdp_profile.py step complete.")

    cleanup()


if __name__ == "__main__":
    main()
