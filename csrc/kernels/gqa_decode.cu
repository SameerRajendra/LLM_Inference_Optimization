/*
 * gqa_decode.cu  —  Split-KV (FlashDecoding-style) GQA decode kernel
 *
 * Layout:
 *   Q   [B, Hq,  D]        fp16   — decode step (seq_len = 1)
 *   K   [B, N,   Hkv, D]   fp16
 *   V   [B, N,   Hkv, D]   fp16
 *   Out [B, Hq,  D]        fp16
 *
 * Why split-KV:
 *   The previous version launched grid = (Hq, B) = 32 blocks for Llama-3.1-8B
 *   decode (batch 1). On a 132-SM H100 that leaves ~76% of the GPU idle, and
 *   each block serially walks the ENTIRE KV sequence for one head — so decode
 *   latency scales ~O(N) with almost no parallelism. That is why it measured
 *   ~3.6x SLOWER than FlashAttention.
 *
 *   FlashDecoding fix: partition the KV sequence into `num_splits` chunks and
 *   launch grid = (Hq, num_splits, B). Each block attends only its chunk and
 *   emits a PARTIAL result — the unnormalized output plus the online-softmax
 *   state (running max m and running sum l). A second `combine` kernel merges
 *   the partials per head with the exact log-sum-exp rescale. This fills all
 *   SMs and cuts each block's serial work to O(N / num_splits).
 *
 *   Pass 1 (gqa_decode_splitkv_kernel):  grid (Hq, num_splits, B)
 *   Pass 2 (gqa_combine_kernel):         grid (Hq, B)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <algorithm>

#define HEAD_DIM      128
#define TILE_SIZE     128
#define BLOCK_THREADS 128
// Shared-memory row stride for the K/V tiles.
//
// With stride == HEAD_DIM (128 halfs = 256 B) the QK dot product below reads
// tile_K[tid][d]: byte offset tid*256 + 2d, so bank = (tid*64 + d/2) % 32.
// tid*64 % 32 == 0 for EVERY tid, i.e. all 32 lanes of a warp land on the same
// bank and each load serialises into 32 transactions — on all 64 iterations of
// the dot product.
//
// Padding by 2 halfs (stride 130 -> 260 B) makes bank = (tid + d/2) % 32, which
// is one distinct bank per lane: conflict-free. 260 B stays 4-byte aligned, so
// the half2 loads remain legal.
#define SMEM_STRIDE   (HEAD_DIM + 2)
#define WARP_SIZE     32
#define FULL_MASK     0xffffffff
#define NUM_WARPS     (BLOCK_THREADS / WARP_SIZE)   // 4

// ─── warp / block reductions ─────────────────────────────────────────────────
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

__device__ __forceinline__ float block_reduce_max(float val, float* smem_warp) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    val = warp_reduce_max(val);
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? smem_warp[threadIdx.x] : -FLT_MAX;
    if (warp_id == 0) val = warp_reduce_max(val);
    if (warp_id == 0 && lane_id == 0) smem_warp[0] = val;
    __syncthreads();
    return smem_warp[0];
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

// ─── Pass 1: per-split partial attention ─────────────────────────────────────
// Each block handles one (q_head, split, b). Thread tid owns output channel tid
// AND computes the score for token tid within each 128-token tile (TILE == D).
__global__ void gqa_decode_splitkv_kernel(
    const __half* __restrict__ Q,          // [B, Hq, D]
    const __half* __restrict__ K,          // [B, N, Hkv, D]
    const __half* __restrict__ V,          // [B, N, Hkv, D]
    float*        __restrict__ O_partial,  // [B, Hq, S, D]  unnormalized
    float*        __restrict__ m_partial,  // [B, Hq, S]     running max
    float*        __restrict__ l_partial,  // [B, Hq, S]     running sum
    int B, int N, int Hq, int Hkv, int D,
    int num_splits, int split_len, float scale)
{
    int q_head = blockIdx.x;
    int split  = blockIdx.y;
    int b      = blockIdx.z;
    int tid    = threadIdx.x;   // 0..127 (channel index)

    int group_size = Hq / Hkv;
    int kv_head    = q_head / group_size;

    int split_start = split * split_len;
    int split_end   = min(split_start + split_len, N);

    int base_hs = (b * Hq + q_head) * num_splits + split;

    // Empty split (can happen only if num_splits over-provisioned): neutral partial.
    if (split_start >= split_end) {
        O_partial[base_hs * D + tid] = 0.f;
        if (tid == 0) { m_partial[base_hs] = -FLT_MAX; l_partial[base_hs] = 0.f; }
        return;
    }

    __shared__ __half tile_Q[HEAD_DIM];
    __shared__ float  scores[TILE_SIZE];
    __shared__ float  warp_buf[NUM_WARPS];

    extern __shared__ __half smem[];
    __half (*tile_K)[SMEM_STRIDE] = reinterpret_cast<__half(*)[SMEM_STRIDE]>(smem);
    __half (*tile_V)[SMEM_STRIDE] = reinterpret_cast<__half(*)[SMEM_STRIDE]>(smem + TILE_SIZE * SMEM_STRIDE);

    // Load Q once into shared memory.
    tile_Q[tid] = Q[b * Hq * D + q_head * D + tid];
    __syncthreads();

    float running_max = -FLT_MAX;
    float running_sum = 0.f;
    float acc         = 0.f;   // unnormalized output for channel tid

    int num_tiles = (split_end - split_start + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int tile_start  = split_start + t * TILE_SIZE;
        int tile_tokens = min(TILE_SIZE, split_end - tile_start);

        // Cooperative coalesced load of this tile's K and V into shared memory.
        // Natural [tok][dim] layout for both — V accum reads tile_V[t][tid]
        // (one row per iter across threads) which is at most a 2-way bank
        // conflict, versus the 32-way conflict of the old transposed layout.
        for (int i = tid; i < TILE_SIZE * HEAD_DIM; i += BLOCK_THREADS) {
            int tok_local = i / HEAD_DIM;
            int dim       = i % HEAD_DIM;
            int tok_glob  = tile_start + tok_local;
            bool valid = (tok_local < tile_tokens);
            int off = b * N * Hkv * D + tok_glob * Hkv * D + kv_head * D + dim;
            tile_K[tok_local][dim] = valid ? K[off] : __float2half(0.f);
            tile_V[tok_local][dim] = valid ? V[off] : __float2half(0.f);
        }
        __syncthreads();

        // Score for token tid (half2-vectorized dot product).
        float score = -FLT_MAX;
        if (tid < tile_tokens) {
            float dot = 0.f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 2) {
                half2 q2 = *reinterpret_cast<const half2*>(&tile_Q[d]);
                half2 k2 = *reinterpret_cast<const half2*>(&tile_K[tid][d]);
                float2 qf = __half22float2(q2);
                float2 kf = __half22float2(k2);
                dot += qf.x * kf.x + qf.y * kf.y;
            }
            score = dot * scale;
        }
        // Reduce straight from the register — the old code round-tripped this
        // through shared memory and cost an extra barrier for no reason.
        float tile_max = block_reduce_max(score, warp_buf);
        __syncthreads();

        float new_max = fmaxf(running_max, tile_max);
        float rescale = expf(running_max - new_max);
        running_max   = new_max;

        float exp_s = (tid < tile_tokens) ? expf(score - running_max) : 0.f;
        scores[tid] = exp_s;
        __syncthreads();

        float tile_sum = block_reduce_sum(exp_s, warp_buf);
        __syncthreads();

        running_sum = running_sum * rescale + tile_sum;

        acc *= rescale;
        #pragma unroll 8
        for (int tt = 0; tt < tile_tokens; tt++)
            acc += scores[tt] * __half2float(tile_V[tt][tid]);
        __syncthreads();
    }

    // Emit UNNORMALIZED partials — the combine kernel does the final divide.
    O_partial[base_hs * D + tid] = acc;
    if (tid == 0) {
        m_partial[base_hs] = running_max;
        l_partial[base_hs] = running_sum;
    }
}

// ─── Pass 2: combine partials across splits ──────────────────────────────────
// grid (Hq, B), block D. Merges num_splits partials per head with the exact
// log-sum-exp rescale:  O = sum_s e^{m_s - M} O_s  /  sum_s e^{m_s - M} l_s.
__global__ void gqa_combine_kernel(
    const float*  __restrict__ O_partial,  // [B, Hq, S, D]
    const float*  __restrict__ m_partial,  // [B, Hq, S]
    const float*  __restrict__ l_partial,  // [B, Hq, S]
    __half*       __restrict__ Out,        // [B, Hq, D]
    int B, int Hq, int D, int num_splits)
{
    int q_head = blockIdx.x;
    int b      = blockIdx.y;
    int tid    = threadIdx.x;   // channel

    int hb = b * Hq + q_head;
    const float* mp = m_partial + hb * num_splits;
    const float* lp = l_partial + hb * num_splits;
    const float* Op = O_partial + hb * num_splits * D;

    float gmax = -FLT_MAX;
    for (int s = 0; s < num_splits; s++) gmax = fmaxf(gmax, mp[s]);

    float acc = 0.f, denom = 0.f;
    for (int s = 0; s < num_splits; s++) {
        float m = mp[s];
        if (m == -FLT_MAX) continue;          // skip empty split
        float w = __expf(m - gmax);
        acc   += w * Op[s * D + tid];
        denom += w * lp[s];
    }

    float inv = (denom > 1e-9f) ? (1.f / denom) : 0.f;
    Out[hb * D + tid] = __float2half_rn(acc * inv);
}

// ─── host launcher ───────────────────────────────────────────────────────────
torch::Tensor launch_fused_gqa(
    torch::Tensor Q,    // [B, Hq, D]      fp16
    torch::Tensor K,    // [B, N, Hkv, D]  fp16
    torch::Tensor V,    // [B, N, Hkv, D]  fp16
    double scale)
{
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
    TORCH_CHECK(D == HEAD_DIM, "D=", D, " — HEAD_DIM must be 128");
    TORCH_CHECK(K.size(3) == D && V.size(3) == D,
                "K/V head_dim must match Q head_dim=", D);
    TORCH_CHECK(V.size(1) == N && V.size(2) == Hkv,
                "V shape must be [B, N, Hkv, D] matching K");
    TORCH_CHECK(Hq % Hkv == 0, "Hq=", Hq, " must be divisible by Hkv=", Hkv);
    TORCH_CHECK(D % 2 == 0, "D must be even for half2 vectorization");

    // ── choose num_splits: enough to fill the GPU AND keep each split small ──
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                           Q.device().index());
    if (sm_count <= 0) sm_count = 108;

    const int SPLIT_TOKENS = 512;   // target tokens per split
    const int MAX_SPLITS   = 256;   // caps partial-buffer size and combine cost

    int splits_by_tokens = (N + SPLIT_TOKENS - 1) / SPLIT_TOKENS;
    int splits_by_occ    = (2 * sm_count + Hq * B - 1) / (Hq * B);  // ~2x SMs
    int num_splits = std::max(splits_by_tokens, splits_by_occ);
    num_splits = std::max(1, std::min(num_splits, MAX_SPLITS));

    int split_len = (N + num_splits - 1) / num_splits;   // tokens per split (ceil)
    num_splits    = (N + split_len - 1) / split_len;     // trim to what N needs

    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto O_partial = torch::empty({B, Hq, num_splits, D}, f32);
    auto m_partial = torch::empty({B, Hq, num_splits},    f32);
    auto l_partial = torch::empty({B, Hq, num_splits},    f32);
    auto Out       = torch::empty({B, Hq, D}, Q.options());

    dim3 block(BLOCK_THREADS);
    // Padded stride (see SMEM_STRIDE): 2 * 128 * 130 * 2 B = 65 KB, still one
    // opt-in >48 KB allocation and unchanged blocks-per-SM vs the 64 KB version.
    int  smem = 2 * TILE_SIZE * SMEM_STRIDE * sizeof(__half);
    cudaFuncSetAttribute(gqa_decode_splitkv_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    dim3 gridA(Hq, num_splits, B);
    gqa_decode_splitkv_kernel<<<gridA, block, smem>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        O_partial.data_ptr<float>(),
        m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(),
        B, N, Hq, Hkv, D, num_splits, split_len,
        static_cast<float>(scale));

    dim3 gridB(Hq, B);
    gqa_combine_kernel<<<gridB, block>>>(
        O_partial.data_ptr<float>(),
        m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(),
        reinterpret_cast<__half*>(Out.data_ptr<at::Half>()),
        B, Hq, D, num_splits);

#ifdef DEBUG_KERNELS
    cudaDeviceSynchronize();
#endif
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "gqa_decode split-KV launch failed: ",
                cudaGetErrorString(err));
    return Out;
}
