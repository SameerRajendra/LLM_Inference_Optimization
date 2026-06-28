#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#define WARP_SIZE     32
#define FULL_MASK     0xffffffff
#define MAX_TOP_K     512
#define BLOCK_THREADS 256

// ─── warp-level reductions ─────────────────────────────────────────────────
__device__ __forceinline__ float warp_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return __shfl_sync(FULL_MASK, val, 0);
}

__device__ __forceinline__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return __shfl_sync(FULL_MASK, val, 0);
}

// ─── block-level max reduction (copied from gqa_decode.cu) ──────────────────
static __forceinline__ __device__ float block_reduce_max(float val, float* smem_warp) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    val = warp_max(val);
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? smem_warp[threadIdx.x] : -FLT_MAX;
    if (warp_id == 0) val = warp_max(val);
    return __shfl_sync(FULL_MASK, val, 0);
}

// ─── block-level sum reduction (copied from gqa_decode.cu) ──────────────────
static __forceinline__ __device__ float block_reduce_sum(float val, float* smem_warp) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    val = warp_sum(val);
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? smem_warp[threadIdx.x] : 0.f;
    if (warp_id == 0) val = warp_sum(val);
    return __shfl_sync(FULL_MASK, val, 0);
}

// ─── online softmax (block-parallel using above reductions) ─────────────────
__device__ void online_softmax(float* scores, int k, float* warp_buf) {
    // max reduction
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        local_max = fmaxf(local_max, scores[i]);
    float global_max = block_reduce_max(local_max, warp_buf);
    __syncthreads();

    // exp + sum
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        scores[i] = expf(scores[i] - global_max);
        local_sum += scores[i];
    }
    float global_sum = block_reduce_sum(local_sum, warp_buf);
    if (global_sum < 1e-9f) global_sum = 1e-9f;
    __syncthreads();

    // normalize
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        scores[i] /= global_sum;
    __syncthreads();
}

// ─── main sparse attention kernel ──────────────────────────────────────────
// Q:   [B, H, 1, D]       fp16
// K:   [B, H, ctx_len, D] fp16
// V:   [B, H, ctx_len, D] fp16
// Out: [B, H, 1, D]       fp16
// top_k_indices: [B, H, top_k] int32 (output)
__global__ void sparse_kv_attn_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ Out,
    float*        __restrict__ scores_buf,      // [B, H, ctx_len] global
    int32_t*      __restrict__ top_k_indices,   // [B, H, top_k] global
    int B, int H, int ctx_len, int D, int top_k,
    float scale)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;  // 8

    // Pointers for this (batch, head)
    const __half* q = Q + (b * H + h) * D;
    const __half* k = K + (b * H + h) * ctx_len * D;
    const __half* v = V + (b * H + h) * ctx_len * D;
    __half*       o = Out + (b * H + h) * D;
    float*        head_scores = scores_buf + (b * H + h) * ctx_len;
    int32_t*      head_topk_idx = top_k_indices + (b * H + h) * top_k;

    int actual_k = min(top_k, ctx_len);
    int local_k = (actual_k + num_warps - 1) / num_warps;  // per-warp budget

    // ── Shared memory layout ───────────────────────────────────────────────
    // warp_buf[num_warps] for block reductions
    // warp_topk_scores[num_warps * local_k] + warp_topk_indices[...]
    // s_topk_scores[actual_k] + s_topk_indices[actual_k]
    // s_out_buf[D]
    extern __shared__ char smem[];
    float* warp_buf = reinterpret_cast<float*>(smem);
    float* warp_topk_scores = warp_buf + num_warps;
    int*   warp_topk_indices = reinterpret_cast<int*>(warp_topk_scores + num_warps * local_k);
    float* s_topk_scores = reinterpret_cast<float*>(warp_topk_indices + num_warps * local_k);
    int*   s_topk_indices = reinterpret_cast<int*>(s_topk_scores + actual_k);
    float* s_out_buf = reinterpret_cast<float*>(s_topk_indices + actual_k);

    // ── Step 1: QK^T scores (parallel across warps) ─────────────────────────
    for (int tok = warp_id * WARP_SIZE + lane_id; tok < ctx_len; tok += num_warps * WARP_SIZE) {
        float dot = 0.f;
        for (int d = 0; d < D; d++)
            dot += __half2float(q[d]) * __half2float(k[tok * D + d]);
        head_scores[tok] = dot * scale;
    }
    __syncthreads();

    // ── Step 2: Top-k via warp-local heaps + merge ──────────────────────────
    // Initialize local heap with -inf
    for (int i = lane_id; i < local_k; i += WARP_SIZE) {
        warp_topk_scores[warp_id * local_k + i] = -FLT_MAX;
        warp_topk_indices[warp_id * local_k + i] = -1;
    }
    __syncthreads();

    // Strided scan: each thread processes multiple tokens, inserts into sorted local array
    for (int tok = warp_id * WARP_SIZE + lane_id; tok < ctx_len; tok += num_warps * WARP_SIZE) {
        float score = head_scores[tok];
        for (int i = 0; i < local_k; i++) {
            if (score > warp_topk_scores[warp_id * local_k + i]) {
                for (int j = local_k - 1; j > i; j--) {
                    warp_topk_scores[warp_id * local_k + j] = warp_topk_scores[warp_id * local_k + j - 1];
                    warp_topk_indices[warp_id * local_k + j] = warp_topk_indices[warp_id * local_k + j - 1];
                }
                warp_topk_scores[warp_id * local_k + i] = score;
                warp_topk_indices[warp_id * local_k + i] = tok;
                break;
            }
        }
    }
    __syncthreads();

    // Warp 0 merges all warp-level top-k lists (k-way merge)
    if (warp_id == 0) {
        float merged_scores[512];
        int merged_indices[512];
        int merge_pos[8] = {0};
        int warp_sizes[8];
        for (int w = 0; w < num_warps; w++) warp_sizes[w] = min(local_k, actual_k);

        for (int pick = 0; pick < actual_k; pick++) {
            float best = -FLT_MAX;
            int best_w = -1;
            for (int w = 0; w < num_warps; w++) {
                int pos = merge_pos[w];
                if (pos < warp_sizes[w]) {
                    float s = warp_topk_scores[w * local_k + pos];
                    if (s > best) {
                        best = s;
                        best_w = w;
                    }
                }
            }
            if (best_w == -1) break;
            merged_scores[pick] = best;
            merged_indices[pick] = warp_topk_indices[best_w * local_k + merge_pos[best_w]];
            merge_pos[best_w]++;
        }

        // Write to shared top-k arrays
        for (int i = lane_id; i < actual_k; i += WARP_SIZE) {
            s_topk_scores[i] = merged_scores[i];
            s_topk_indices[i] = merged_indices[i];
            head_topk_idx[i] = merged_indices[i];
        }
    }
    __syncthreads();

    // ── Step 3: Softmax over top-k ─────────────────────────────────────────
    online_softmax(s_topk_scores, actual_k, warp_buf);
    __syncthreads();

    // ── Step 4: Weighted sum over top-k V rows ─────────────────────────────
    for (int d = tid; d < D; d += blockDim.x)
        s_out_buf[d] = 0.f;
    __syncthreads();

    for (int i = 0; i < actual_k; i++) {
        int tok = s_topk_indices[i];
        float w = s_topk_scores[i];
        for (int d = tid; d < D; d += blockDim.x)
            s_out_buf[d] += w * __half2float(v[tok * D + d]);
        __syncthreads();
    }

    // Write output
    for (int d = tid; d < D; d += blockDim.x)
        o[d] = __float2half_rn(s_out_buf[d]);
}

// ─── Host launcher (SAME SIGNATURE AS ORIGINAL) ────────────────────────────
torch::Tensor kv_evict_quant_forward(
    torch::Tensor Q,       // [B, H, 1, D] fp16
    torch::Tensor K,       // [B, H, ctx_len, D] fp16
    torch::Tensor V,       // [B, H, ctx_len, D] fp16
    int top_k,
    bool use_int8)         // kept for API compatibility; ignored when false
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Expected fp16 Q");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "Expected fp16 K");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "Expected fp16 V");
    TORCH_CHECK(use_int8 == false, "INT8 path not enabled in this build");

    int B       = Q.size(0);
    int H       = Q.size(1);
    int ctx_len = K.size(2);
    int D       = Q.size(3);

    TORCH_CHECK(D <= 4096, "D exceeds max supported");
    TORCH_CHECK(top_k <= MAX_TOP_K, "top_k exceeds MAX_TOP_K");

    float scale = 1.f / sqrtf(static_cast<float>(D));

    auto Out = torch::zeros({B, H, 1, D}, Q.options());
    auto top_k_idx = torch::zeros({B, H, top_k},
                     torch::dtype(torch::kInt32).device(Q.device()));
    auto scores_buf = torch::empty({B, H, ctx_len},
                       torch::dtype(torch::kFloat32).device(Q.device()));

    dim3 grid(B, H);
    dim3 block(BLOCK_THREADS);  // 256 threads = 8 warps

    // Shared memory: 
    // warp_buf[num_warps] + warp_topk_scores[num_warps*local_k] + warp_topk_indices[num_warps*local_k]
    // + s_topk_scores[top_k] + s_topk_indices[top_k] + s_out_buf[D]
    int num_warps = BLOCK_THREADS / WARP_SIZE;
    int local_k = (top_k + num_warps - 1) / num_warps;
    size_t smem_bytes = 
        num_warps * sizeof(float) +                               // warp_buf
        num_warps * local_k * sizeof(float) +                     // warp_topk_scores
        num_warps * local_k * sizeof(int) +                       // warp_topk_indices
        top_k * sizeof(float) + top_k * sizeof(int) +             // s_topk
        D * sizeof(float);                                        // s_out_buf

    sparse_kv_attn_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<__half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Out.data_ptr<at::Half>()),
        scores_buf.data_ptr<float>(),
        top_k_idx.data_ptr<int32_t>(),
        B, H, ctx_len, D, top_k, scale);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "sparse_kv_attn_kernel launch failed: ",
                cudaGetErrorString(err));
    return Out;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("kv_evict_quant_forward", &kv_evict_quant_forward,
//           "Sparse KV Attention Forward (top-k eviction)");
// }