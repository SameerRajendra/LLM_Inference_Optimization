#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE      32
#define FULL_MASK      0xffffffff
#define MAX_BLOCKS_TOPK 128
#define MAX_HEAD_DIM   1024

// ─── warp reductions ────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return __shfl_sync(FULL_MASK, val, 0);
}

__device__ __forceinline__ float warp_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return __shfl_sync(FULL_MASK, val, 0);
}

// ─── main kernel ────────────────────────────────────────────────────────────
__global__ void block_sparse_attn_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ Out,
    int32_t*      __restrict__ block_idx_out,
    float*        __restrict__ block_scores_buf,
    int B, int H, int T, int D,
    int block_size,
    int top_k_blocks,
    float scale)
{
    int b   = blockIdx.x;
    int h   = blockIdx.y;
    int tid = threadIdx.x;

    const __half* q = Q + (b * H + h) * D;
    const __half* k = K + (b * H + h) * T * D;
    const __half* v = V + (b * H + h) * T * D;
    __half*       o = Out + (b * H + h) * D;

    int num_blocks  = (T + block_size - 1) / block_size;
    float* block_scores = block_scores_buf + (b * H + h) * num_blocks;

    // ── Step 1: score each block (half2 vectorised) ──────────────────────────
    for (int blk = tid; blk < num_blocks; blk += WARP_SIZE) {
        int start = blk * block_size;
        int end   = min(start + block_size, T);
        int count = end - start;

        float accum = 0.f;
        for (int tok = start; tok < end; tok++) {
            float dot = 0.f;

            // half2 vectorised dot product — 2× throughput
            // Requires D to be even (guaranteed by transformer head dims)
            const __half* krow = k + tok * D;
            int d = 0;
            for (; d + 1 < D; d += 2) {
                half2 q2 = *reinterpret_cast<const half2*>(&q[d]);
                half2 k2 = *reinterpret_cast<const half2*>(&krow[d]);
                float2 qf = __half22float2(q2);
                float2 kf = __half22float2(k2);
                dot += qf.x * kf.x + qf.y * kf.y;
            }
            // scalar tail if D is odd
            if (d < D)
                dot += __half2float(q[d]) * __half2float(krow[d]);

            accum += dot * scale;
        }
        block_scores[blk] = (count > 0) ? (accum / count) : -FLT_MAX;
    }
    __syncthreads();

    // ── Step 2: top-k selection (thread 0, min-heap replace) ────────────────
    // replaced broken insertion sort with correct min-heap replacement.
    // Maintains a max-heap of size actual_k by replacing the minimum element
    // whenever a higher score is found — O(num_blocks * actual_k) but correct.
    __shared__ float top_scores[MAX_BLOCKS_TOPK];
    __shared__ int   top_blocks[MAX_BLOCKS_TOPK];

    if (tid == 0) {
        int actual_k = min(top_k_blocks, num_blocks);

        // Initialise heap slots to -inf
        for (int i = 0; i < actual_k; i++) {
            top_scores[i] = -FLT_MAX;
            top_blocks[i] = -1;
        }

        // Min-heap replacement: O(num_blocks * actual_k)
        for (int blk = 0; blk < num_blocks; blk++) {
            float s = block_scores[blk];

            // Find the current minimum in top_scores
            int min_idx = 0;
            for (int i = 1; i < actual_k; i++)
                if (top_scores[i] < top_scores[min_idx]) min_idx = i;

            // Replace minimum if current score is larger
            if (s > top_scores[min_idx]) {
                top_scores[min_idx] = s;
                top_blocks[min_idx] = blk;
            }
        }

        // Write selected block indices to global output
        int32_t* idx_out = block_idx_out + (b * H + h) * top_k_blocks;
        for (int i = 0; i < actual_k; i++)   idx_out[i] = (int32_t)top_blocks[i];
        for (int i = actual_k; i < top_k_blocks; i++) idx_out[i] = -1;
    }
    __syncthreads();

    // ── Step 3: softmax over top-k block scores ──────────────────────────────
    int actual_k = min(top_k_blocks, num_blocks);

    // Max reduction across warp
    float local_max = -FLT_MAX;
    for (int i = tid; i < actual_k; i += WARP_SIZE)
        local_max = fmaxf(local_max, top_scores[i]);
    float global_max = warp_max(local_max);

    // Exp + partial sum
    float local_sum = 0.f;
    for (int i = tid; i < actual_k; i += WARP_SIZE) {
        top_scores[i] = expf(top_scores[i] - global_max);
        local_sum += top_scores[i];
    }
    float global_sum = warp_sum(local_sum);
    if (global_sum < 1e-9f) global_sum = 1e-9f;

    // Normalise
    for (int i = tid; i < actual_k; i += WARP_SIZE)
        top_scores[i] /= global_sum;

    //  __syncthreads() BEFORE V accumulation reads top_scores.
    // Without this, thread 0 reads top_scores[1..] before threads 1..
    // finish writing their normalised values — data race.
    __syncthreads();

    // ── Step 4: accumulate weighted V ───────────────────────────────────────
    __shared__ float out_buf[MAX_HEAD_DIM];
    for (int d = tid; d < D; d += WARP_SIZE)
        out_buf[d] = 0.f;
    __syncthreads();

    for (int i = 0; i < actual_k; i++) {
        int blk = top_blocks[i];
        if (blk < 0) continue;                // guard: skip unfilled heap slots

        float w_blk  = top_scores[i];
        int   start  = blk * block_size;
        int   end    = min(start + block_size, T);
        int   count  = end - start;
        float token_w = (count > 0) ? (w_blk / count) : 0.f;

        for (int tok = start; tok < end; tok++) {
            for (int d = tid; d < D; d += WARP_SIZE)
                out_buf[d] += token_w * __half2float(v[tok * D + d]);
        }
        // NOTE: no __syncthreads() inside loop — each thread owns
        // disjoint d-slots so no cross-thread write conflict per block.
    }
    __syncthreads();   //  one barrier after full accumulation, not per-block

    // Write output
    for (int d = tid; d < D; d += WARP_SIZE)
        o[d] = __float2half(out_buf[d]);
}

// ─── host launcher ──────────────────────────────────────────────────────────
torch::Tensor sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int block_size,
    int top_k_blocks)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be fp16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be fp16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be fp16");

    // FIX P1: validate decode-step shape
    TORCH_CHECK(Q.size(2) == 1,
                "sparse_attention_forward: Q must be decode-step shape [B,H,1,D]");
    TORCH_CHECK(K.size(0) == Q.size(0) && K.size(1) == Q.size(1),
                "K batch/head dims must match Q");
    TORCH_CHECK(V.size(2) == K.size(2) && V.size(3) == K.size(3),
                "V shape must match K");
    TORCH_CHECK(top_k_blocks > 0, "top_k_blocks must be positive");
    TORCH_CHECK(block_size  > 0, "block_size must be positive");

    int B  = Q.size(0);
    int H  = Q.size(1);
    int T  = K.size(2);
    int D  = Q.size(3);

    TORCH_CHECK(D <= MAX_HEAD_DIM,
                "D=", D, " exceeds MAX_HEAD_DIM=", MAX_HEAD_DIM);
    TORCH_CHECK(top_k_blocks <= MAX_BLOCKS_TOPK,
                "top_k_blocks=", top_k_blocks, " exceeds MAX_BLOCKS_TOPK=", MAX_BLOCKS_TOPK);

    auto Out = torch::zeros({B, H, 1, D}, Q.options());
    int num_blocks_kv = (T + block_size - 1) / block_size;

    auto block_idx = torch::full(
        {B, H, top_k_blocks}, -1,
        torch::dtype(torch::kInt32).device(Q.device()));
    auto block_scores = torch::empty(
        {B, H, num_blocks_kv},
        torch::dtype(torch::kFloat32).device(Q.device()));

    float scale = 1.0f / sqrtf((float)D);

    dim3 grid(B, H);
    dim3 block(WARP_SIZE);

    block_sparse_attn_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Out.data_ptr<at::Half>()),
        block_idx.data_ptr<int32_t>(),
        block_scores.data_ptr<float>(),
        B, H, T, D, block_size, top_k_blocks, scale);

    //  synchronize in debug builds to catch runtime errors
#ifdef DEBUG_KERNELS
    cudaDeviceSynchronize();
#endif
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "block_sparse_attn_kernel launch failed: ",
                cudaGetErrorString(err));

    return Out;
}
