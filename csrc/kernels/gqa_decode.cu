/*
 * gqa_decode.cu  —  FlashAttention-2 online softmax GQA decode kernel
 *
 * Layout:
 *   Q   [B, Hq,  D]        fp16   — decode step (seq_len = 1)
 *   K   [B, N,  Hkv, D]   fp16
 *   V   [B, N,  Hkv, D]   fp16
 *   Out [B, Hq,  D]        fp16
 *
 * Tiling:
 *   grid  (Hq, B)
 *   block BLOCK_THREADS = HEAD_DIM = 128 threads
 *   Each thread owns one output channel (tid = channel index)
 *   Tiles over KV sequence in chunks of TILE_SIZE tokens
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#define HEAD_DIM      128
#define TILE_SIZE     128
#define BLOCK_THREADS 128
#define WARP_SIZE     32
#define FULL_MASK     0xffffffff
#define NUM_WARPS     (BLOCK_THREADS / WARP_SIZE)   // 4

// ─── warp reductions ─────────────────────────────────────────────────────────
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

// ─── block reductions ────────────────────────────────────────────────────────

__device__ __forceinline__ float block_reduce_max(float val, float* smem_warp) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Step 1: reduce within each warp
    val = warp_reduce_max(val);
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();

    // Step 2: warp 0 reduces across warp results
    val = (threadIdx.x < NUM_WARPS) ? smem_warp[threadIdx.x] : -FLT_MAX;
    if (warp_id == 0) val = warp_reduce_max(val);

    // Step 3: write result to smem[0] so ALL warps can read it
    if (warp_id == 0 && lane_id == 0) smem_warp[0] = val;
    __syncthreads();

    return smem_warp[0];   // every thread reads the same value
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem_warp) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < NUM_WARPS) ? smem_warp[threadIdx.x] : 0.f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    if (warp_id == 0 && lane_id == 0) smem_warp[0] = val;
    __syncthreads();

    return smem_warp[0];
}

// ─── main kernel ─────────────────────────────────────────────────────────────
extern "C" __global__ void gqa_decode_kernel(
    const __half* __restrict__ Q,    // [B, Hq, D]
    const __half* __restrict__ K,    // [B, N, Hkv, D]
    const __half* __restrict__ V,    // [B, N, Hkv, D]
    __half*       __restrict__ Out,  // [B, Hq, D]
    int B, int N, int Hq, int Hkv, int D,
    float scale)
{
    int b      = blockIdx.y;
    int q_head = blockIdx.x;
    int tid    = threadIdx.x;   // 0..127  (== channel index)

    int group_size = Hq / Hkv;
    int kv_head    = q_head / group_size;

    // ── shared memory layout ──────────────────────────────────────────────────
    // tile_Q  [HEAD_DIM]              — Q loaded once, reused every tile
    // tile_K  [TILE_SIZE][HEAD_DIM]   — current KV tile
    // tile_V  [HEAD_DIM][TILE_SIZE]   — TRANSPOSED vs original
    //                                   FIX [P2-6]: row access in V accum
    //                                   eliminates shared memory bank conflicts
    // scores  [TILE_SIZE]             — exp(score - max) per token in tile
    // warp_buf[NUM_WARPS]             — block reduction scratch
    //

    __shared__ __half tile_Q[HEAD_DIM];
    __shared__ float  scores[TILE_SIZE];
    __shared__ float  warp_buf[NUM_WARPS];

    // ── 2. Large arrays become dynamic ────────────────────────────────────
    extern __shared__ __half dynamic_smem[];
    __half (*tile_K)[HEAD_DIM] = reinterpret_cast<__half(*)[HEAD_DIM]>(dynamic_smem);
    __half (*tile_V)[TILE_SIZE] = reinterpret_cast<__half(*)[TILE_SIZE]>(dynamic_smem + TILE_SIZE * HEAD_DIM);

    // ── FIX [P0-1]: load Q into shared memory ONCE ───────────────────────────
    // Original loaded Q from global memory inside every tile iteration
    // (HEAD_DIM reads × num_tiles × per thread = 128 × 500 = 64,000 at 64K).
    // Loading once into shared memory costs HEAD_DIM reads total.
    tile_Q[tid] = Q[b * Hq * D + q_head * D + tid];
    __syncthreads();

    // ── online softmax accumulators ───────────────────────────────────────────
    float running_max = -FLT_MAX;
    float running_sum = 0.f;
    float acc         = 0.f;   // output channel = tid, accumulated across tiles

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start  = tile * TILE_SIZE;
        int tile_tokens = min(TILE_SIZE, N - tile_start);

        // ── load KV tile cooperatively ────────────────────────────────────────
        // BLOCK_THREADS=128 threads load TILE_SIZE*HEAD_DIM=16384 fp16 values
        // Each thread loads 128 elements — coalesced global memory access
        //
        // tile_K stored as [tok][dim] — coalesced load, used for dot product
        // tile_V stored as [dim][tok] — FIX [P2-6]: transposed so V accum
        //   reads tile_V[tid][t] (row access) instead of tile_V[t][tid]
        //   (column access) — eliminates shared memory bank conflicts
        int total_elems = TILE_SIZE * HEAD_DIM;
        for (int i = tid; i < total_elems; i += BLOCK_THREADS) {
            int tok_local  = i / HEAD_DIM;
            int dim        = i % HEAD_DIM;
            int tok_global = tile_start + tok_local;

            bool valid = (tok_global < N);
            int kv_offset = b * N * Hkv * D
                          + tok_global * Hkv * D
                          + kv_head * D + dim;

            __half kval = valid ? K[kv_offset] : __float2half(0.f);
            __half vval = valid ? V[kv_offset] : __float2half(0.f);

            tile_K[tok_local][dim] = kval;
            tile_V[dim][tok_local] = vval;   // transposed store
        }
        __syncthreads();

        // ── compute dot(Q, K[tid]) for token tid in this tile ─────────────────
       
        float score = -FLT_MAX;
        if (tid < tile_tokens) {
            float dot = 0.f;
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; d += 2) {
                half2 q2 = *reinterpret_cast<const half2*>(&tile_Q[d]);
                half2 k2 = *reinterpret_cast<const half2*>(&tile_K[tid][d]);
                float2 qf = __half22float2(q2);
                float2 kf = __half22float2(k2);
                dot += qf.x * kf.x + qf.y * kf.y;
            }
            score = dot * scale;
        }
        scores[tid] = score;
        __syncthreads();

        // ── online softmax update ─────────────────────────────────────────────
        // block_reduce_max now correct for all warps —
        // warps 1-3 previously received -FLT_MAX, causing them to never
        // update running_max and accumulate V with wrong weights.
        float tile_max = block_reduce_max(scores[tid], warp_buf);
        __syncthreads();

        float new_max = fmaxf(running_max, tile_max);
        float rescale = expf(running_max - new_max);   // 1.0 if max unchanged
        running_max   = new_max;

        float exp_s    = (tid < tile_tokens) ? expf(score - running_max) : 0.f;
        scores[tid]    = exp_s;
        __syncthreads();

        float tile_sum = block_reduce_sum(exp_s, warp_buf);
        __syncthreads();

        running_sum = running_sum * rescale + tile_sum;

        // ── accumulate weighted V for channel tid ─────────────────────────────
        // tile_V[tid][t] is row access — no bank conflict
        // Original tile_V[t][tid] was column access — bank conflict every iter
        acc *= rescale;
        #pragma unroll 8
        for (int t = 0; t < tile_tokens; t++)
            acc += scores[t] * __half2float(tile_V[tid][t]);

        __syncthreads();
    }

    // ── normalize and write output ────────────────────────────────────────────
    float inv = (running_sum > 1e-9f) ? (1.f / running_sum) : 0.f;
    Out[b * Hq * D + q_head * D + tid] = __float2half_rn(acc * inv);
}

// ─── host launcher ───────────────────────────────────────────────────────────
torch::Tensor launch_fused_gqa(
    torch::Tensor Q,    // [B, Hq, D]      fp16
    torch::Tensor K,    // [B, N, Hkv, D]  fp16
    torch::Tensor V,    // [B, N, Hkv, D]  fp16
    double scale)
{
    // full input validation
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "Q/K/V must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat16 &&
                K.dtype() == torch::kFloat16 &&
                V.dtype() == torch::kFloat16,
                "all inputs must be fp16");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "all inputs must be contiguous");

    int B   = Q.size(0);
    int Hq  = Q.size(1);
    int D   = Q.size(2);
    int N   = K.size(1);
    int Hkv = K.size(2);

    TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0),
                "batch size mismatch across Q/K/V");
    TORCH_CHECK(D == HEAD_DIM,
                "D=", D, " — HEAD_DIM must be 128");
    TORCH_CHECK(K.size(3) == D && V.size(3) == D,
                "K/V head_dim must match Q head_dim=", D);
    TORCH_CHECK(V.size(1) == N && V.size(2) == Hkv,
                "V shape must be [B, N, Hkv, D] matching K");
    TORCH_CHECK(Hq % Hkv == 0,
                "Hq=", Hq, " must be divisible by Hkv=", Hkv);
    TORCH_CHECK(D % 2 == 0,
                "D must be even for half2 vectorization");

    auto Out = torch::zeros({B, Hq, D}, Q.options());

    dim3 grid(Hq, B);
    dim3 block(BLOCK_THREADS);

    // Shared memory: tile_Q + tile_K + tile_V + scores + warp_buf
    // = HEAD_DIM*2 + TILE_SIZE*HEAD_DIM*2 + HEAD_DIM*TILE_SIZE*2
    //   + TILE_SIZE*4 + NUM_WARPS*4
    // = 256 + 32768 + 32768 + 512 + 16 = 66,320 bytes (~65 KB)
    // A100 supports 164KB shared memory per SM with opt-in —
    // default limit is 48KB. For large HEAD_DIM*TILE_SIZE tiles,
    // request extended shared memory:
    int dynamic_smem_bytes = 2 * TILE_SIZE * HEAD_DIM * sizeof(__half);

    // 2. Request permission from CUDA to exceed the standard 48KB limit
    cudaFuncSetAttribute(
        gqa_decode_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_smem_bytes);

    // 3. Launch the kernel, passing the dynamic bytes as the 3rd argument
    gqa_decode_kernel<<<grid, block, dynamic_smem_bytes>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Out.data_ptr<at::Half>()),
        B, N, Hq, Hkv, D,
        static_cast<float>(scale));

#ifdef DEBUG_KERNELS
    cudaDeviceSynchronize();
#endif
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "gqa_decode_kernel launch failed: ",
                cudaGetErrorString(err));
    return Out;
}

