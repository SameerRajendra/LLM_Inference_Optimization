#include <torch/extension.h>

torch::Tensor kv_evict_quant_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int top_k, bool use_int8);

// gqa_decode.cu
torch::Tensor launch_fused_gqa(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    double scale);

torch::Tensor launch_fused_gqa_v4(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    double scale);

std::vector<int64_t> gqa_kernel_info(int64_t hq, int64_t hkv);

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("kv_evict_quant_forward", &kv_evict_quant_forward,
//           "Sparse KV eviction + attention output (CUDA)");
//     m.def("fused_gqa", &launch_fused_gqa,
//           "Fused GQA tiled decode kernel (Hopper sm_90)");
// }

torch::Tensor sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int block_size,
    int top_k_blocks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attention_forward", &sparse_attention_forward, "Block sparse attention forward");
    m.def("kv_evict_quant_forward", &kv_evict_quant_forward, "KV evict quant forward");
    m.def("fused_gqa", &launch_fused_gqa,
          "Fused GQA tiled decode kernel, v3 split-KV (Hopper sm_90)");
    m.def("fused_gqa_v4", &launch_fused_gqa_v4,
          "GQA decode v4: one block per KV head, one warp per query head");
    m.def("gqa_kernel_info", &gqa_kernel_info,
          "[regs3, smem3, blocks_per_sm3, threads3, "
          "regs4, smem4, blocks_per_sm4, threads4, sm_count]");
}