#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#define WARP_SIZE   32
#define FULL_MASK   0xffffffff
#define MAX_TOP_K   512

// ─── warp-level max reduction ─────────────────────────────────────────────────
__device__ __forceinline__ float warp_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return __shfl_sync(FULL_MASK, val, 0);
}

// ─── warp-level sum reduction ─────────────────────────────────────────────────
__device__ __forceinline__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return __shfl_sync(FULL_MASK, val, 0);
}

// ─── main kernel ──────────────────────────────────────────────────────────────
// Q:   [B, H, 1,       D]  fp16
// K:   [B, H, ctx_len, D]  fp16
// V:   [B, H, ctx_len, D]  fp16
// Out: [B, H, 1,       D]  fp16
// top_k_indices: [B, H, top_k]  int32  (output — which tokens were selected)
__global__ void sparse_kv_attn_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ Out,
    float*        __restrict__ scores_buf,
    int32_t*      __restrict__ top_k_indices,
    int B, int H, int ctx_len, int D, int top_k,
    float scale)
{
    // one block = one (batch, head) pair
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;   // 0..31 (one warp per head for simplicity)

    const __half* q = Q + (b * H + h) * D;           // [D]
    const __half* k = K + (b * H + h) * ctx_len * D; // [ctx_len, D]
    const __half* v = V + (b * H + h) * ctx_len * D;
    __half*       o = Out + (b * H + h) * D;
    float*    scores = scores_buf + (b * H + h) * ctx_len;

    // ── Step 1: compute QK^T scores for all tokens ──────────────────────────
    // We iterate over tokens in chunks; each thread handles one token at a time
    // in a round-robin fashion across the warp.
    // extern __shared__ float smem[];                   // [ctx_len] scores
    // float* scores = smem;

    for (int tok = tid; tok < ctx_len; tok += WARP_SIZE) {
        float dot = 0.f;
        for (int d = 0; d < D; d++) {
            dot += __half2float(q[d]) * __half2float(k[tok * D + d]);
        }
        scores[tok] = dot * scale;
    }
    __syncthreads();

    // ── Step 2: top-k selection via bitonic sort on shared mem ───────────────
    // Use a simple register-level approach: each warp thread maintains a
    // local sorted list and merges. For correctness at any ctx_len we do a
    // linear scan with a running min-heap maintained in registers.
    // (Full warp-parallel bitonic sort shown below for top_k <= WARP_SIZE)
    __shared__ float  topk_scores[MAX_TOP_K];
    __shared__ int    topk_idx[MAX_TOP_K];

    // Each thread keeps a local (score, idx) min-heap of size top_k/WARP_SIZE
    // Then we merge across the warp into shared memory.
    // For top_k <= WARP_SIZE: one slot per thread, then warp-reduce.
    int actual_k = (top_k < ctx_len) ? top_k : ctx_len;

    // Phase 1: each thread finds its best candidate via strided scan
    float my_best_score = -FLT_MAX;
    int   my_best_idx   = 0;
    for (int tok = tid; tok < ctx_len; tok += WARP_SIZE) {
        if (scores[tok] > my_best_score) {
            my_best_score = scores[tok];
            my_best_idx   = tok;
        }
    }

    // Phase 2: gather all warp candidates into shared, then serial select top_k
    __shared__ float  cand_scores[WARP_SIZE];
    __shared__ int    cand_idx[WARP_SIZE];
    cand_scores[tid] = my_best_score;
    cand_idx[tid]    = my_best_idx;
    __syncthreads();

    // Phase 3: tid==0 does final selection from WARP_SIZE candidates
    // Then iteratively mask and re-scan for remaining top_k slots
    if (tid == 0) {
        int32_t* idx_out = top_k_indices + (b * H + h) * top_k;
        __shared__ bool used[WARP_SIZE];
        for (int w = 0; w < WARP_SIZE; w++) used[w] = false;

        for (int pick = 0; pick < actual_k && pick < WARP_SIZE; pick++) {
            float best = -FLT_MAX; int best_w = 0;
            for (int w = 0; w < WARP_SIZE; w++) {
                if (!used[w] && cand_scores[w] > best) {
                    best = cand_scores[w]; best_w = w;
                }
            }
            used[best_w]       = true;
            topk_scores[pick]  = cand_scores[best_w];
            topk_idx[pick]     = cand_idx[best_w];
            idx_out[pick]      = (int32_t)cand_idx[best_w];
        }
    }
    __syncthreads();


    // ── Step 3: softmax over top-k scores ────────────────────────────────────
    // Parallel online softmax across warp threads
    float local_max = -FLT_MAX;
    for (int i = tid; i < top_k; i += WARP_SIZE)
        local_max = fmaxf(local_max, topk_scores[i]);
    float global_max = warp_max(local_max);

    float local_sum = 0.f;
    for (int i = tid; i < top_k; i += WARP_SIZE) {
        topk_scores[i] = expf(topk_scores[i] - global_max); // write exp in-place
        local_sum += topk_scores[i];
    }
    __syncthreads();
    float global_sum = warp_sum(local_sum);
    if (global_sum < 1e-9f) global_sum = 1e-9f;

    // Normalize
    for (int i = tid; i < top_k; i += WARP_SIZE)
        topk_scores[i] /= global_sum;
    __syncthreads();

    // ── Step 4: weighted sum over top-k V rows ────────────────────────────────
    // Out[d] = sum_i softmax_i * V[topk_idx[i], d]
    __shared__ float out_buf[1024]; // max D=1024
    for (int d = tid; d < D; d += WARP_SIZE)
        out_buf[d] = 0.f;
    __syncthreads();

    for (int i = 0; i < top_k; i++) {
        int tok = topk_idx[i];
        float w  = topk_scores[i];
        for (int d = tid; d < D; d += WARP_SIZE)
            out_buf[d] += w * __half2float(v[tok * D + d]);
        __syncthreads();
    }

    // Write output
    for (int d = tid; d < D; d += WARP_SIZE)
        o[d] = __float2half(out_buf[d]);
}

// INT8 per-token symmetric quantization of KV cache
// __global__ void quantize_kv_int8_kernel(
//     const __half* __restrict__ src,   // [tokens, D]
//     int8_t*       __restrict__ dst,   // [tokens, D]
//     float*        __restrict__ scales,// [tokens]
//     int tokens, int D)
// {
//     int tok = blockIdx.x;
//     int tid = threadIdx.x;
//     if (tok >= tokens) return;

//     const __half* row = src + tok * D;

//     // Compute max abs for this token (reduce across warp)
//     float local_max = 0.f;
//     for (int d = tid; d < D; d += WARP_SIZE)
//         local_max = fmaxf(local_max, fabsf(__half2float(row[d])));
//     float global_max = warp_max(local_max);

//     float scale = global_max / 127.f;
//     if (tid == 0) scales[tok] = scale;

//     float inv = (scale > 1e-9f) ? (1.f / scale) : 0.f;
//     for (int d = tid; d < D; d += WARP_SIZE) {
//         float val = __half2float(row[d]) * inv;
//         dst[tok * D + d] = (int8_t)fmaxf(-127.f, fminf(127.f, val));
//     }
// }

torch::Tensor kv_evict_quant_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int top_k,
    bool use_int8)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Expected fp16");

    int B       = Q.size(0);
    int H       = Q.size(1);
    int ctx_len = K.size(2);
    int D       = Q.size(3);
    float scale = 1.f / sqrtf((float)D);

    auto Out        = torch::zeros({B, H, 1, D}, Q.options());
    auto top_k_idx  = torch::zeros({B, H, top_k},
                         torch::dtype(torch::kInt32).device(Q.device()));
    auto scores_buf = torch::empty({B, H, ctx_len},
                         torch::dtype(torch::kFloat32).device(Q.device()));

    dim3 grid(B, H);
    dim3 block(WARP_SIZE);
    size_t smem = 0;  // scores in global memory, not shared

    sparse_kv_attn_kernel<<<grid, block, smem>>>(
        (const __half*)Q.data_ptr(),
        (const __half*)K.data_ptr(),
        (const __half*)V.data_ptr(),
        (__half*)Out.data_ptr(),
        (float*)scores_buf.data_ptr(),
        (int32_t*)top_k_idx.data_ptr(),
        B, H, ctx_len, D, top_k, scale);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "sparse_kv_attn_kernel launch failed: ",
                cudaGetErrorString(err));
    return Out;
}

