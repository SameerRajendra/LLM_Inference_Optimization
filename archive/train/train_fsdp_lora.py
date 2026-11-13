import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.cuda.nvtx import range_push, range_pop

def setup_distributed():
    """Initializes the multi-GPU process group utilizing the NCCL communication fabric."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

class LoRALinear(nn.Module):
    """Custom low-rank parameterized adapter block injected into dense layers."""
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: int = 32):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        # LoRA adapter tracking matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        
        # Freeze base parameters to track gradients strictly on the adapters
        self.base_layer.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_outputs = self.base_layer(x)
        lora_outputs = (x @ self.lora_A) @ self.lora_B * self.scaling
        return base_outputs + lora_outputs

class LlamaStyleDecoderLayer(nn.Module):
    """A multi-stage transformer layer to simulate real structural workloads."""
    def __init__(self, dim: int = 8192):
        super().__init__()
        # Self-Attention projections with low-rank adapters attached
        self.q_proj = LoRALinear(dim, dim)
        self.k_proj = LoRALinear(dim, dim)
        self.v_proj = LoRALinear(dim, dim)
        self.o_proj = LoRALinear(dim, dim)
        
        # SwiGLU MLP Block architecture
        self.gate_proj = nn.Linear(dim, 28672, bias=False)
        self.up_proj = nn.Linear(dim, 28672, bias=False)
        self.down_proj = nn.Linear(28672, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Approximate multi-head computation tracks
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn_out = self.o_proj(q + k + v)
        residual_attn = x + attn_out
        
        # SwiGLU structural sequence calculation
        mlp_out = self.down_proj(self.act(self.gate_proj(residual_attn)) * self.up_proj(residual_attn))
        return residual_attn + mlp_out

def main():
    local_rank = setup_distributed()
    if local_rank == 0:
        print("[*] NCCL Fabric connected. Instantiating 4-layer Llama model stack...")

    device = torch.device(f"cuda:{local_rank}")
    
    # Construct a deep sequence stack directly on the local accelerator target
    model = nn.Sequential(*[LlamaStyleDecoderLayer() for _ in range(4)]).to(device)
    
    # CRITICAL: Define an explicit auto-wrap policy targeting the Decoder Layer boundaries.
    # This shards each layer independently, allowing backward-pass compute steps to 
    # overlap with NCCL communications for adjacent layers.
    layer_wrap_policy = ModuleWrapPolicy({LlamaStyleDecoderLayer})
    
    sharded_model = FSDP(
        model,
        auto_wrap_policy=layer_wrap_policy,
        device_id=local_rank,
        use_orig_params=True
    )
    
    # Configure optimizer to track only the un-frozen trainable adapter weights
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sharded_model.parameters()), lr=1e-4)
    criterion = nn.MSELoss()

    # High-density input allocations to saturate the H200 execution lanes
    inputs = torch.randn(8, 2048, 8192, device=device)
    targets = torch.randn(8, 2048, 8192, device=device)

    if local_rank == 0:
        print("[*] Cold execution warmup iterations enqueued...")
        
    # Warm up compilation blocks to keep initialization tasks out of our profile
    for _ in range(2):
        outputs = sharded_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    dist.barrier()

    if local_rank == 0:
        print("[*] Warmup finalized. Commencing NVTX-profiled core iteration block...")

    # Core Profiled Execution Cycle
    range_push("End_to_End_FSDP_Optimization_Step")
    
    range_push("Forward_Pass_Compute")
    outputs = sharded_model(inputs)
    loss = criterion(outputs, targets)
    torch.cuda.synchronize()
    range_pop() # End Forward_Pass_Compute

    range_push("Backward_Pass_Compute_and_Comm_Overlap")
    loss.backward()
    torch.cuda.synchronize()
    range_pop() # End Backward_Pass_Compute_and_Comm_Overlap

    range_push("Optimizer_Parameter_Updates")
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    range_pop() # End Optimizer_Parameter_Updates
    
    range_pop() # End End_to_End_FSDP_Optimization_Step

    if local_rank == 0:
        print("[+] Optimization loop complete. Systems profile file written out successfully.")

    cleanup()

if __name__ == "__main__":
    main()
