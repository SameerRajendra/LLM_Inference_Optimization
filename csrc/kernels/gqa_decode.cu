/*
 * Algorithm: FlashAttention-2 online softmax, single pass over KV cache
 * Layout:    Q [B, Hq, D]   K/V [B, N, Hkv, D]   fp16
 * Tiling:    one warp (32 threads) per query head
 *            each thread owns D/32 = 4 channels (for D=128)
 *            tiles over KV sequence in chunks of TILE_SIZE tokens
 *
 * Key techniques:
 *   - __half2 vectorized loads (2x throughput vs scalar __half)
 *   - warp-level dot product via __shfl_xor_sync reduction
 *   - online softmax: running (max, sum, acc) — no recompute pass
 *   - GQA: kv_head = q_head / (Hq / Hkv)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#define HEAD_DIM     128
#define TILE_SIZE    128    // tokens per tile — one per thread
#define BLOCK_THREADS 128   // threads per block = TILE_SIZE
#define WARP_SIZE    32
#define FULL_MASK    0xffffffff
#define NUM_WARPS    (BLOCK_THREADS / WARP_SIZE)  // 4

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(FULL_MASK, v, mask);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, mask));
    return v;
}

// Block-level max reduction across NUM_WARPS warps
__device__ float block_reduce_max(float val, float* smem_warp) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    val = warp_reduce_max(val);
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? smem_warp[threadIdx.x] : -FLT_MAX;
    if (warp_id == 0) val = warp_reduce_max(val);
    return __shfl_sync(FULL_MASK, val, 0);
}

// Block-level sum reduction
__device__ float block_reduce_sum(float val, float* smem_warp) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    val = warp_reduce_sum(val);
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? smem_warp[threadIdx.x] : 0.f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return __shfl_sync(FULL_MASK, val, 0);
}

__global__ void gqa_decode_kernel(
    const __half* __restrict__ Q,    // [B, Hq, D]
    const __half* __restrict__ K,    // [B, N, Hkv, D]
    const __half* __restrict__ V,    // [B, N, Hkv, D]
    __half*       __restrict__ Out,  // [B, Hq, D]
    int B, int N, int Hq, int Hkv, int D,
    float scale)
{
    int b      = blockIdx.y;
    int q_head = blockIdx.x;
    int tid    = threadIdx.x;   // 0..127

    int group_size = Hq / Hkv;
    int kv_head    = q_head / group_size;

    // ── load Q into registers: each thread owns 1 channel ────────────────────
    // D=128, BLOCK_THREADS=128 → 1 float per thread
    float q_val = __half2float(Q[b * Hq * D + q_head * D + tid]);

    // ── shared memory layout ──────────────────────────────────────────────────
    // tile_K/V: [TILE_SIZE, D] — loaded cooperatively
    // scores:   [TILE_SIZE]    — one score per thread
    // warp_buf: [NUM_WARPS]    — for block reductions
    __shared__ __half tile_K[TILE_SIZE][HEAD_DIM];
    __shared__ __half tile_V[TILE_SIZE][HEAD_DIM];
    __shared__ float  scores[TILE_SIZE];
    __shared__ float  warp_buf[NUM_WARPS];
    __shared__ float  out_buf[HEAD_DIM];

    // ── online softmax accumulators ───────────────────────────────────────────
    float running_max = -FLT_MAX;
    float running_sum = 0.f;
    // output accumulator: each thread accumulates its own channel
    float acc = 0.f;

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start  = tile * TILE_SIZE;
        int tile_tokens = min(TILE_SIZE, N - tile_start);

        // ── load KV tile: BLOCK_THREADS threads load TILE_SIZE*D values ──────
        // Total elements = 128*128 = 16384 halfs
        // Each thread loads 16384/128 = 128 elements
        int total_elems = TILE_SIZE * HEAD_DIM;
        for (int i = tid; i < total_elems; i += BLOCK_THREADS) {
            int tok_local  = i / HEAD_DIM;
            int dim        = i % HEAD_DIM;
            int tok_global = tile_start + tok_local;
            int kv_offset  = b * N * Hkv * D
                           + tok_global * Hkv * D
                           + kv_head * D + dim;
            tile_K[tok_local][dim] = (tok_global < N) ?
                K[kv_offset] : __float2half(0.f);
            tile_V[tok_local][dim] = (tok_global < N) ?
                V[kv_offset] : __float2half(0.f);
        }
        __syncthreads();

        // ── each thread computes dot(Q, K[tid]) ──────────────────────────────
        // tid < tile_tokens: this thread is responsible for token tid in tile
        float score = -FLT_MAX;
        if (tid < tile_tokens) {
            float dot = 0.f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++)
                dot += __half2float(Q[b * Hq * D + q_head * D + d])
                     * __half2float(tile_K[tid][d]);
            score = dot * scale;
            scores[tid] = score;
        } else {
            scores[tid] = -FLT_MAX;
        }
        __syncthreads();

        // ── block-level max for online softmax ────────────────────────────────
        float tile_max = block_reduce_max(scores[tid], warp_buf);
        __syncthreads();

        // ── compute exp(score - new_max), update running stats ────────────────
        float new_max  = fmaxf(running_max, tile_max);
        float rescale  = expf(running_max - new_max);
        running_max    = new_max;

        float exp_s = (tid < tile_tokens) ?
            expf(scores[tid] - running_max) : 0.f;
        scores[tid] = exp_s;   // reuse scores[] for exp values
        __syncthreads();

        float tile_sum = block_reduce_sum(exp_s, warp_buf);
        __syncthreads();

        running_sum = running_sum * rescale + tile_sum;

        // ── accumulate weighted V: each thread owns channel = tid ─────────────
        acc *= rescale;
        for (int t = 0; t < tile_tokens; t++)
            acc += scores[t] * __half2float(tile_V[t][tid]);
        __syncthreads();
    }

    // ── normalize and write output ────────────────────────────────────────────
    float inv = (running_sum > 1e-9f) ? (1.f / running_sum) : 0.f;
    Out[b * Hq * D + q_head * D + tid] = __float2half(acc * inv);
}

// ── C++ launcher ─────────────────────────────────────────────────────────────
torch::Tensor launch_fused_gqa(
    torch::Tensor Q,    // [B, Hq, D]      fp16
    torch::Tensor K,    // [B, N, Hkv, D]  fp16
    torch::Tensor V,    // [B, N, Hkv, D]  fp16
    double scale)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Expected fp16");

    int B   = Q.size(0);
    int Hq  = Q.size(1);
    int D   = Q.size(2);
    int N   = K.size(1);
    int Hkv = K.size(2);

    TORCH_CHECK(D == HEAD_DIM,  "HEAD_DIM must be 128");
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");

    auto Out = torch::zeros({B, Hq, D}, Q.options());

    dim3 grid(Hq, B);
    dim3 block(BLOCK_THREADS);

    gqa_decode_kernel<<<grid, block>>>(
        (const __half*)Q.data_ptr(),
        (const __half*)K.data_ptr(),
        (const __half*)V.data_ptr(),
        (__half*)Out.data_ptr(),
        B, N, Hq, Hkv, D,
        static_cast<float>(scale));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "gqa_decode_kernel launch failed: ",
                cudaGetErrorString(err));
    return Out;
}