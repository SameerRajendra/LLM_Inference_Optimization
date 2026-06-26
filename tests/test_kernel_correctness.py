import torch
from sparse_kv._C import kv_evict_quant_forward
from sparse_kv.attention import sparse_attention

B, H, N, D, TOP_K = 1, 32, 4096, 128, 32
torch.manual_seed(42)

Q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

cuda_out = kv_evict_quant_forward(Q, K, V, TOP_K, False)
ref_out  = sparse_attention(Q, K, V, TOP_K, sink_tokens=0)

diff = (cuda_out - ref_out).abs().max().item()
print(f"Max diff: {diff:.5f}")
assert diff < 0.15, f"❌ Kernel diverged: max diff = {diff:.5f}"
print("✅ Kernel output matches reference")