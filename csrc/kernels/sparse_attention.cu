#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define MAX_BLOCKS_TOPK 128
#define MAX_HEAD_DIM 1024

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

__global__ void block_sparse_attn_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ Out,
    int32_t* __restrict__ block_idx_out,
    float* __restrict__ block_scores_buf,
    int B, int H, int T, int D,
    int block_size,
    int top_k_blocks,
    float scale)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    const __half* q = Q + ((b * H + h) * D);
    const __half* k = K + ((b * H + h) * T * D);
    const __half* v = V + ((b * H + h) * T * D);
    __half* o = Out + ((b * H + h) * D);

    int num_blocks = (T + block_size - 1) / block_size;
    float* block_scores = block_scores_buf + ((b * H + h) * num_blocks);

    for (int blk = tid; blk < num_blocks; blk += WARP_SIZE) {
        int start = blk * block_size;
        int end = start + block_size;
        if (end > T) end = T;

        float accum = 0.f;
        int count = end - start;

        for (int tok = start; tok < end; tok++) {
            float dot = 0.f;
            for (int d = 0; d < D; d++) {
                dot += __half2float(q[d]) * __half2float(k[tok * D + d]);
            }
            accum += dot * scale;
        }
        block_scores[blk] = (count > 0) ? (accum / count) : -FLT_MAX;
    }
    __syncthreads();

    __shared__ float top_scores[MAX_BLOCKS_TOPK];
    __shared__ int top_blocks[MAX_BLOCKS_TOPK];

    if (tid == 0) {
        int actual_k = top_k_blocks < num_blocks ? top_k_blocks : num_blocks;

        for (int i = 0; i < actual_k; i++) {
            top_scores[i] = -FLT_MAX;
            top_blocks[i] = -1;
        }

        for (int blk = 0; blk < num_blocks; blk++) {
            float s = block_scores[blk];
            int insert = -1;

            for (int i = 0; i < actual_k; i++) {
                if (s > top_scores[i]) {
                    insert = i;
                    break;
                }
            }

            if (insert >= 0) {
                for (int j = actual_k - 1; j > insert; j--) {
                    top_scores[j] = top_scores[j - 1];
                    top_blocks[j] = top_blocks[j - 1];
                }
                top_scores[insert] = s;
                top_blocks[insert] = blk;
            }
        }

        int32_t* idx_out = block_idx_out + ((b * H + h) * top_k_blocks);
        for (int i = 0; i < actual_k; i++) idx_out[i] = top_blocks[i];
        for (int i = actual_k; i < top_k_blocks; i++) idx_out[i] = -1;
    }
    __syncthreads();

    int actual_k = top_k_blocks < num_blocks ? top_k_blocks : num_blocks;

    float local_max = -FLT_MAX;
    for (int i = tid; i < actual_k; i += WARP_SIZE) {
        local_max = fmaxf(local_max, top_scores[i]);
    }
    float global_max = warp_max(local_max);

    float local_sum = 0.f;
    for (int i = tid; i < actual_k; i += WARP_SIZE) {
        top_scores[i] = expf(top_scores[i] - global_max);
        local_sum += top_scores[i];
    }
    __syncthreads();

    float global_sum = warp_sum(local_sum);
    if (global_sum < 1e-9f) global_sum = 1e-9f;

    for (int i = tid; i < actual_k; i += WARP_SIZE) {
        top_scores[i] /= global_sum;
    }
    __syncthreads();

    __shared__ float out_buf[MAX_HEAD_DIM];
    for (int d = tid; d < D; d += WARP_SIZE) {
        out_buf[d] = 0.f;
    }
    __syncthreads();

    for (int i = 0; i < actual_k; i++) {
        int blk = top_blocks[i];
        if (blk < 0) continue;

        float w_blk = top_scores[i];
        int start = blk * block_size;
        int end = start + block_size;
        if (end > T) end = T;
        int count = end - start;
        float token_w = (count > 0) ? (w_blk / count) : 0.f;

        for (int tok = start; tok < end; tok++) {
            for (int d = tid; d < D; d += WARP_SIZE) {
                out_buf[d] += token_w * __half2float(v[tok * D + d]);
            }
        }
        __syncthreads();
    }

    for (int d = tid; d < D; d += WARP_SIZE) {
        o[d] = __float2half(out_buf[d]);
    }
}

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

    int B = Q.size(0);
    int H = Q.size(1);
    int T = K.size(2);
    int D = Q.size(3);

    TORCH_CHECK(D <= MAX_HEAD_DIM, "D exceeds MAX_HEAD_DIM");
    TORCH_CHECK(top_k_blocks <= MAX_BLOCKS_TOPK, "top_k_blocks exceeds MAX_BLOCKS_TOPK");

    auto Out = torch::zeros({B, H, 1, D}, Q.options());
    int num_blocks = (T + block_size - 1) / block_size;

    auto block_idx = torch::full(
        {B, H, top_k_blocks},
        -1,
        torch::dtype(torch::kInt32).device(Q.device()));

    auto block_scores = torch::empty(
        {B, H, num_blocks},
        torch::dtype(torch::kFloat32).device(Q.device()));

    float scale = 1.0f / sqrtf((float)D);

    dim3 grid(B, H);
    dim3 block(WARP_SIZE);

    block_sparse_attn_kernel<<<grid, block>>>(
        (const __half*)Q.data_ptr(),
        (const __half*)K.data_ptr(),
        (const __half*)V.data_ptr(),
        (__half*)Out.data_ptr(),
        (int32_t*)block_idx.data_ptr(),
        (float*)block_scores.data_ptr(),
        B, H, T, D, block_size, top_k_blocks, scale);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "block_sparse_attn_kernel launch failed: ",
                cudaGetErrorString(err));

    return Out;
}