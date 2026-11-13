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

torch::Tensor launch_fused_gqa_v4_fp8(
    torch::Tensor Q, torch::Tensor Kq, torch::Tensor Vq,
    torch::Tensor k_scale, torch::Tensor v_scale, double scale);

std::vector<int64_t> gqa_kernel_info(int64_t hq, int64_t hkv);

// gqa_decode_tc.cu
torch::Tensor launch_fused_gqa_tc(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    double scale, bool causal);

torch::Tensor launch_fused_gqa_tc_fp8(
    torch::Tensor Q, torch::Tensor Kq, torch::Tensor Vq,
    torch::Tensor k_scale, torch::Tensor v_scale, double scale, bool causal);

std::vector<int64_t> gqa_tc_kernel_info(int64_t head_dim, bool fp8);

// kv_fp8.cu
std::vector<torch::Tensor> quantize_kv_fp8(torch::Tensor K, torch::Tensor V);

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
    m.def("quantize_kv_fp8", &quantize_kv_fp8,
          "Quantize fp16 K/V to e4m3 with per-(page, kv_head) scales "
          "-> {Kq, Vq, k_scale, v_scale}");
    m.def("fused_gqa_v4_fp8", &launch_fused_gqa_v4_fp8,
          "GQA decode v4 reading an e4m3 KV cache (half the HBM traffic)");
    m.def("fused_gqa_tc", &launch_fused_gqa_tc,
          "GQA decode tensor-core + cp.async pipeline (fp16, supports S_q >= 1)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"),
          py::arg("causal") = false);
    m.def("fused_gqa_tc_fp8", &launch_fused_gqa_tc_fp8,
          "GQA decode tensor-core + cp.async pipeline (e4m3 KV, supports S_q >= 1)",
          py::arg("Q"), py::arg("Kq"), py::arg("Vq"),
          py::arg("k_scale"), py::arg("v_scale"), py::arg("scale"),
          py::arg("causal") = false);
    m.def("gqa_tc_kernel_info", &gqa_tc_kernel_info,
          "[regs, local_bytes, smem_bytes, blocks_per_sm, threads]");
}