#include <torch/extension.h>

torch::Tensor kv_evict_quant_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int top_k, bool use_int8);

// gqa_decode.cu
torch::Tensor launch_fused_gqa(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    double scale);

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
          "Fused GQA tiled decode kernel (Hopper sm_90)");
}