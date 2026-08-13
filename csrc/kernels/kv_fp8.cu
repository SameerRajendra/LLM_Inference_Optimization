/*
 * kv_fp8.cu — FP8 (e4m3) KV-cache quantisation
 *
 * Why: at long context the decode step is bound by KV-cache bytes, not FLOPs.
 * Storing K/V as e4m3 instead of fp16 halves both the HBM traffic per decode
 * step AND the resident cache, so a GPU holds twice the concurrent sequences.
 * For Llama-3.1-8B at 64K that is 8.4 GB -> 4.2 GB per sequence, i.e. ~7 -> ~14
 * concurrent sequences on an 80 GB H100.
 *
 * Granularity: one scale per (page of KV_PAGE_TOKENS tokens, kv_head), computed
 * separately for K and V. Per-page rather than per-tensor because attention
 * activations drift in magnitude along the sequence, and a single global scale
 * would let one outlier page crush the resolution of every other page. Keeping
 * the page equal to the decode kernel's tile means the decode kernel reads one
 * scale per tile rather than doing a per-token lookup.
 *
 * Format: e4m3 (not e5m2). KV values are narrow-range and precision-sensitive,
 * so the extra mantissa bit matters more than the extra exponent range; e5m2 is
 * the better choice for gradients, not for activations.
 *
 * Layout:
 *   K, V      [B, N, Hkv, D]      fp16   (input)
 *   Kq, Vq    [B, N, Hkv, D]      uint8  (e4m3 bits)
 *   k_scale   [B, num_pages, Hkv] fp32
 *   v_scale   [B, num_pages, Hkv] fp32
 */
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <vector>

#include "fp8_common.cuh"

#define QUANT_THREADS 256
#define QUANT_WARPS   (QUANT_THREADS / 32)
#define FULL_MASK_Q   0xffffffff

__device__ __forceinline__ float warp_max_f(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK_Q, v, m));
    return v;
}

__device__ __forceinline__ float block_max_f(float v, float* sm) {
    const int w = threadIdx.x / 32;
    const int l = threadIdx.x % 32;
    v = warp_max_f(v);
    if (l == 0) sm[w] = v;
    __syncthreads();
    v = (threadIdx.x < QUANT_WARPS) ? sm[threadIdx.x] : 0.f;
    if (w == 0) v = warp_max_f(v);
    if (threadIdx.x == 0) sm[0] = v;
    __syncthreads();
    const float r = sm[0];
    __syncthreads();
    return r;
}

// One block per (page, kv_head, batch). Two passes over the page: absmax, then
// quantise. The page is ~16 KB so it stays hot in L1/L2 between the passes.
__global__ void quantize_kv_fp8_kernel(
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __nv_fp8_storage_t* __restrict__ Kq,
    __nv_fp8_storage_t* __restrict__ Vq,
    float* __restrict__ k_scale,
    float* __restrict__ v_scale,
    int B, int N, int Hkv, int D, int num_pages)
{
    const int page = blockIdx.x;
    const int head = blockIdx.y;
    const int b    = blockIdx.z;
    const int tid  = threadIdx.x;

    const int tok0 = page * KV_PAGE_TOKENS;
    const int tok1 = min(tok0 + KV_PAGE_TOKENS, N);
    if (tok0 >= N) return;
    const int n_elem = (tok1 - tok0) * D;

    __shared__ float sm[QUANT_WARPS];

    float kmax = 0.f, vmax = 0.f;
    for (int i = tid; i < n_elem; i += QUANT_THREADS) {
        const int off = ((b * N + tok0 + i / D) * Hkv + head) * D + (i % D);
        kmax = fmaxf(kmax, fabsf(__half2float(K[off])));
        vmax = fmaxf(vmax, fabsf(__half2float(V[off])));
    }
    const float kabs = block_max_f(kmax, sm);
    const float vabs = block_max_f(vmax, sm);

    // An all-zero page would otherwise divide by zero; scale 1 is harmless
    // because every value it encodes is 0.
    const float ks = (kabs > 0.f) ? (kabs / E4M3_MAX) : 1.f;
    const float vs = (vabs > 0.f) ? (vabs / E4M3_MAX) : 1.f;
    if (tid == 0) {
        k_scale[(b * num_pages + page) * Hkv + head] = ks;
        v_scale[(b * num_pages + page) * Hkv + head] = vs;
    }

    const float inv_k = 1.f / ks;
    const float inv_v = 1.f / vs;
    for (int i = tid; i < n_elem; i += QUANT_THREADS) {
        const int off = ((b * N + tok0 + i / D) * Hkv + head) * D + (i % D);
        Kq[off] = __nv_cvt_float_to_fp8(
            __half2float(K[off]) * inv_k, __NV_SATFINITE, __NV_E4M3);
        Vq[off] = __nv_cvt_float_to_fp8(
            __half2float(V[off]) * inv_v, __NV_SATFINITE, __NV_E4M3);
    }
}

// Returns {Kq, Vq, k_scale, v_scale}.
std::vector<torch::Tensor> quantize_kv_fp8(torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(K.is_cuda() && V.is_cuda(), "K/V must be CUDA tensors");
    TORCH_CHECK(K.dtype() == torch::kFloat16 && V.dtype() == torch::kFloat16,
                "K/V must be fp16");
    TORCH_CHECK(K.is_contiguous() && V.is_contiguous(), "K/V must be contiguous");
    TORCH_CHECK(K.dim() == 4, "K must be [B, N, Hkv, D]");
    TORCH_CHECK(V.sizes() == K.sizes(), "V must have the same shape as K");

    const int B   = K.size(0);
    const int N   = K.size(1);
    const int Hkv = K.size(2);
    const int D   = K.size(3);
    const int num_pages = (N + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS;

    auto u8  = torch::TensorOptions().dtype(torch::kUInt8).device(K.device());
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(K.device());
    auto Kq = torch::empty({B, N, Hkv, D}, u8);
    auto Vq = torch::empty({B, N, Hkv, D}, u8);
    auto k_scale = torch::empty({B, num_pages, Hkv}, f32);
    auto v_scale = torch::empty({B, num_pages, Hkv}, f32);

    dim3 grid(num_pages, Hkv, B);
    quantize_kv_fp8_kernel<<<grid, QUANT_THREADS>>>(
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__nv_fp8_storage_t*>(Kq.data_ptr<uint8_t>()),
        reinterpret_cast<__nv_fp8_storage_t*>(Vq.data_ptr<uint8_t>()),
        k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
        B, N, Hkv, D, num_pages);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "quantize_kv_fp8 launch failed: ", cudaGetErrorString(err));
    return {Kq, Vq, k_scale, v_scale};
}
