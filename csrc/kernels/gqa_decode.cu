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
#include <cuda_fp8.h>
#include <float.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "fp8_common.cuh"

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

// ═════════════════════════════════════════════════════════════════════════════
// v4: one block per KV HEAD (not per query head), one warp per query head
// ═════════════════════════════════════════════════════════════════════════════
//
// Two measured problems with the v3 kernel above, both provable from its launch
// config rather than from a profiler (this cluster has no Nsight Compute
// counter access):
//
//   1. 4x redundant HBM traffic. v3's grid is (Hq, splits, B) and derives
//      kv_head = q_head / G, so the G=4 query-head blocks of a group each load
//      the SAME K/V tile. At 64K that is 1.02 GB of load requests against
//      262 MB of unique KV.
//
//   2. 18.75% occupancy. v3 uses 2*128*130*2 B = 65 KB of shared per block, so
//      only floor(228/65) = 3 blocks fit per SM; at 128 threads that is 384 of
//      2048 threads = 12 warps, far too few to hide ~500-cycle global latency
//      behind the serial FMA chains in the score and accumulate loops.
//
// v4 fixes both with one restructure: the block is indexed by kv_head, so each
// K/V tile is fetched ONCE and shared by all G query heads (4x less traffic),
// and the tile is halved to 64 tokens so shared drops to ~34 KB and 6 blocks
// fit per SM. One warp owns one query head, which makes the whole online
// softmax a warp-shuffle reduction -- no __syncthreads at all in the softmax,
// down from ~9 barriers per tile to 2.
//
// Work assignment inside a block:
//   warp w        -> query head kv_head*G + w
//   lane l        -> tokens {l, l+32} of the 64-token tile (2 scores)
//   lane l output -> channels {2l, 2l+1} and {2l+64, 2l+65}  (4 of 128)
//
// Shared-memory banking (stride 130 halfs = 260 B, as in v3):
//   score read  tile_K[l][d]     -> bank (l*65 + d/2) % 32   distinct per lane
//   value read  tile_V[t][2l]    -> bank (t*65 + l)   % 32   distinct per lane
// Both conflict-free. Global loads are uint4 (16 B); the shared stores are
// half2 because a 260 B row stride is 4-byte but not 16-byte aligned, and the
// global side is the expensive one.

// Tokens per shared-memory tile.
//
// 64 after a measured negative result. Stage 4a tried 32, which halves the K/V
// shared footprint (35.3 -> 18.2 KB) and doubles residency (6 -> 12 blocks/SM,
// 37.5% -> 75% occupancy). Everything got SLOWER: 64K fp16 0.1397 -> 0.1474 ms,
// 4K fp16 0.0190 -> 0.0253, 64K fp8 0.1251 -> 0.1372. Twice the tiles means
// twice the barriers and loop overhead, and that cost more than the extra warps
// returned.
//
// The useful conclusion is what it rules OUT: this kernel is not occupancy
// -limited, so the ~86 us byte-independent floor at 64K is instruction issue,
// not latency hiding. At ~1700 instructions per lane per tile, 24 warps/SM and
// 4 issue slots/cycle, the issue-rate estimate lands near the observed floor.
// Reducing instruction COUNT is therefore the lever -- i.e. tensor-core MMA,
// where one m16n8k16 does 2048 MACs against 32 for a warp of scalar FMAs.
#define TILE_V4 64
#define VEC_HALF 8                     // uint4 = 8 halfs

// One query-head row against one K row. Q is read broadcast (same address for
// every lane); K is read at tile_K[tok][d] with tok distinct per lane, which
// the padded stride keeps conflict-free.
__device__ __forceinline__ float qk_dot(const __half* __restrict__ qh,
                                        const __half* __restrict__ krow,
                                        float scale) {
    float dot = 0.f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d += 2) {
        const half2 q2 = *reinterpret_cast<const half2*>(&qh[d]);
        const half2 k2 = *reinterpret_cast<const half2*>(&krow[d]);
        const float2 qf = __half22float2(q2);
        const float2 kf = __half22float2(k2);
        dot += qf.x * kf.x + qf.y * kf.y;
    }
    return dot * scale;
}

__global__ void gqa_decode_v4_kernel(
    const __half* __restrict__ Q,          // [B, Hq, D]
    const __half* __restrict__ K,          // [B, N, Hkv, D]
    const __half* __restrict__ V,          // [B, N, Hkv, D]
    float*        __restrict__ O_partial,  // [B, Hq, S, D]  unnormalized
    float*        __restrict__ m_partial,  // [B, Hq, S]
    float*        __restrict__ l_partial,  // [B, Hq, S]
    int B, int N, int Hq, int Hkv, int D, int G,
    int num_splits, int split_len, float scale)
{
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    const int b       = blockIdx.z;
    const int tid     = threadIdx.x;
    const int nthr    = blockDim.x;          // G * 32
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    const int q_head  = kv_head * G + warp_id;

    const int split_start = split * split_len;
    const int split_end   = min(split_start + split_len, N);
    const int base_hs     = (b * Hq + q_head) * num_splits + split;

    extern __shared__ __half smem_v4[];
    __half (*tile_K)[SMEM_STRIDE] =
        reinterpret_cast<__half(*)[SMEM_STRIDE]>(smem_v4);
    __half (*tile_V)[SMEM_STRIDE] =
        reinterpret_cast<__half(*)[SMEM_STRIDE]>(smem_v4 + TILE_V4 * SMEM_STRIDE);
    __half* tile_Q = smem_v4 + 2 * TILE_V4 * SMEM_STRIDE;              // [G][D]
    float*  scores = reinterpret_cast<float*>(tile_Q + G * HEAD_DIM);  // [G][TILE]
    float*  my_scores = scores + warp_id * TILE_V4;

    if (split_start >= split_end) {                    // over-provisioned split
        for (int c = lane; c < D; c += WARP_SIZE)
            O_partial[base_hs * D + c] = 0.f;
        if (lane == 0) { m_partial[base_hs] = -FLT_MAX; l_partial[base_hs] = 0.f; }
        return;
    }

    // Q for every query head of this group, read broadcast-style in the dot.
    for (int i = tid; i < G * HEAD_DIM; i += nthr)
        tile_Q[i] = Q[b * Hq * D + (kv_head * G + i / HEAD_DIM) * D + i % HEAD_DIM];

    float running_max = -FLT_MAX;
    float running_sum = 0.f;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;   // channels 2l,2l+1,2l+64,2l+65

    const int num_tiles = (split_end - split_start + TILE_V4 - 1) / TILE_V4;

    for (int t = 0; t < num_tiles; t++) {
        const int tile_start  = split_start + t * TILE_V4;
        const int tile_tokens = min(TILE_V4, split_end - tile_start);

        __syncthreads();                     // previous tile fully consumed

        // ---- cooperative vectorized load: uint4 from global, half2 to shared ----
        for (int i = tid * VEC_HALF; i < TILE_V4 * HEAD_DIM; i += nthr * VEC_HALF) {
            const int tok_local = i / HEAD_DIM;
            const int dim       = i % HEAD_DIM;
            const int tok_glob  = tile_start + tok_local;
            if (tok_local < tile_tokens) {
                const int off = b * N * Hkv * D + tok_glob * Hkv * D + kv_head * D + dim;
                const uint4 kv = *reinterpret_cast<const uint4*>(K + off);
                const uint4 vv = *reinterpret_cast<const uint4*>(V + off);
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    reinterpret_cast<uint32_t*>(&tile_K[tok_local][dim])[j] =
                        reinterpret_cast<const uint32_t*>(&kv)[j];
                    reinterpret_cast<uint32_t*>(&tile_V[tok_local][dim])[j] =
                        reinterpret_cast<const uint32_t*>(&vv)[j];
                }
            } else {
                #pragma unroll
                for (int j = 0; j < VEC_HALF; j++) {
                    tile_K[tok_local][dim + j] = __float2half(0.f);
                    tile_V[tok_local][dim + j] = __float2half(0.f);
                }
            }
        }
        __syncthreads();

        // ---- scores: lane owns tokens {lane, lane+32}; pure warp reduction ----
        const __half* qh = tile_Q + warp_id * HEAD_DIM;
        const float s0 = (lane < tile_tokens)
                       ? qk_dot(qh, tile_K[lane], scale) : -FLT_MAX;
        const float s1 = (lane + WARP_SIZE < tile_tokens)
                       ? qk_dot(qh, tile_K[lane + WARP_SIZE], scale) : -FLT_MAX;

        const float tile_max = warp_reduce_max(fmaxf(s0, s1));
        const float new_max  = fmaxf(running_max, tile_max);
        const float rescale  = (running_max == -FLT_MAX) ? 0.f
                                                         : __expf(running_max - new_max);
        running_max = new_max;

        const float e0 = (s0 > -FLT_MAX) ? __expf(s0 - running_max) : 0.f;
        const float e1 = (s1 > -FLT_MAX) ? __expf(s1 - running_max) : 0.f;
        my_scores[lane] = e0;
        if (lane + WARP_SIZE < TILE_V4) my_scores[lane + WARP_SIZE] = e1;
        __syncwarp();

        running_sum = running_sum * rescale + warp_reduce_sum(e0 + e1);

        // ---- accumulate PV: half2 reads keep every lane on its own bank ----
        acc0 *= rescale; acc1 *= rescale; acc2 *= rescale; acc3 *= rescale;
        for (int tt = 0; tt < tile_tokens; tt++) {
            const float w = my_scores[tt];
            const half2 va = *reinterpret_cast<const half2*>(&tile_V[tt][2 * lane]);
            const half2 vb = *reinterpret_cast<const half2*>(&tile_V[tt][2 * lane + 64]);
            const float2 fa = __half22float2(va);
            const float2 fb = __half22float2(vb);
            acc0 += w * fa.x; acc1 += w * fa.y;
            acc2 += w * fb.x; acc3 += w * fb.y;
        }
    }

    float* out = O_partial + base_hs * D;
    out[2 * lane]          = acc0;
    out[2 * lane + 1]      = acc1;
    out[2 * lane + 64]     = acc2;
    out[2 * lane + 64 + 1] = acc3;
    if (lane == 0) {
        m_partial[base_hs] = running_max;
        l_partial[base_hs] = running_sum;
    }
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

// ═════════════════════════════════════════════════════════════════════════════
// v4-fp8: identical to v4 except K/V arrive as e4m3 and are dequantised on load
// ═════════════════════════════════════════════════════════════════════════════
//
// The kernel is memory-bound (v4 sits at 56% of HBM peak at 64K), so halving
// the bytes read is worth more than any remaining compute tuning. Only the tile
// load changes: 16 B of global traffic now carries 16 values instead of 8, and
// each is scaled back to fp16 in-register before landing in shared. Everything
// downstream -- qk_dot, the online softmax, the PV accumulation -- is
// byte-identical to v4, which is deliberate: it keeps the A/B honest, since any
// timing difference can only come from the traffic.
//
// One scale per tile, not per token. The launcher forces split_len to a
// multiple of KV_PAGE_TOKENS, so as long as the tile divides the page evenly
// every tile lies wholly inside one page and page = tile_start/KV_PAGE_TOKENS
// is exact. (Tiles LARGER than a page would straddle two scales.)
static_assert(KV_PAGE_TOKENS % TILE_V4 == 0,
              "decode tile must divide the quantisation page evenly, or a tile "
              "straddles two pages and the scale lookup below is wrong");

__global__ void gqa_decode_v4_fp8_kernel(
    const __half*             __restrict__ Q,         // [B, Hq, D]
    const __nv_fp8_storage_t* __restrict__ Kq,        // [B, N, Hkv, D] e4m3
    const __nv_fp8_storage_t* __restrict__ Vq,        // [B, N, Hkv, D] e4m3
    const float*              __restrict__ k_scale,   // [B, pages, Hkv]
    const float*              __restrict__ v_scale,   // [B, pages, Hkv]
    float*                    __restrict__ O_partial,
    float*                    __restrict__ m_partial,
    float*                    __restrict__ l_partial,
    int B, int N, int Hq, int Hkv, int D, int G,
    int num_splits, int split_len, int num_pages, float scale)
{
    const int kv_head = blockIdx.x;
    const int split   = blockIdx.y;
    const int b       = blockIdx.z;
    const int tid     = threadIdx.x;
    const int nthr    = blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    const int q_head      = kv_head * G + warp_id;
    const int split_start = split * split_len;
    const int split_end   = min(split_start + split_len, N);
    const int base_hs     = (b * Hq + q_head) * num_splits + split;

    extern __shared__ __half smem_f8[];
    __half (*tile_K)[SMEM_STRIDE] =
        reinterpret_cast<__half(*)[SMEM_STRIDE]>(smem_f8);
    __half (*tile_V)[SMEM_STRIDE] =
        reinterpret_cast<__half(*)[SMEM_STRIDE]>(smem_f8 + TILE_V4 * SMEM_STRIDE);
    __half* tile_Q = smem_f8 + 2 * TILE_V4 * SMEM_STRIDE;
    float*  scores = reinterpret_cast<float*>(tile_Q + G * HEAD_DIM);
    float*  my_scores = scores + warp_id * TILE_V4;

    if (split_start >= split_end) {
        for (int c = lane; c < D; c += WARP_SIZE)
            O_partial[base_hs * D + c] = 0.f;
        if (lane == 0) { m_partial[base_hs] = -FLT_MAX; l_partial[base_hs] = 0.f; }
        return;
    }

    for (int i = tid; i < G * HEAD_DIM; i += nthr)
        tile_Q[i] = Q[b * Hq * D + (kv_head * G + i / HEAD_DIM) * D + i % HEAD_DIM];

    float running_max = -FLT_MAX;
    float running_sum = 0.f;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    const int num_tiles = (split_end - split_start + TILE_V4 - 1) / TILE_V4;

    for (int t = 0; t < num_tiles; t++) {
        const int tile_start  = split_start + t * TILE_V4;
        const int tile_tokens = min(TILE_V4, split_end - tile_start);

        // split_len is a multiple of KV_PAGE_TOKENS, so this tile is one page.
        const int page = tile_start / KV_PAGE_TOKENS;
        const half2 ks2 = __float2half2_rn(
            k_scale[(b * num_pages + page) * Hkv + kv_head]);
        const half2 vs2 = __float2half2_rn(
            v_scale[(b * num_pages + page) * Hkv + kv_head]);

        __syncthreads();

        // 16 B per thread now carries 16 e4m3 values (vs 8 fp16) -- this is the
        // entire point: same instruction count, half the bytes.
        for (int i = tid * 16; i < TILE_V4 * HEAD_DIM; i += nthr * 16) {
            const int tok_local = i / HEAD_DIM;
            const int dim       = i % HEAD_DIM;
            if (tok_local < tile_tokens) {
                const int off = b * N * Hkv * D + (tile_start + tok_local) * Hkv * D
                              + kv_head * D + dim;
                const uint4 kraw = *reinterpret_cast<const uint4*>(Kq + off);
                const uint4 vraw = *reinterpret_cast<const uint4*>(Vq + off);
                const uint16_t* kp = reinterpret_cast<const uint16_t*>(&kraw);
                const uint16_t* vp = reinterpret_cast<const uint16_t*>(&vraw);
                #pragma unroll
                for (int j = 0; j < 8; j++) {     // 8 pairs = 16 values
                    __half2_raw kh = __nv_cvt_fp8x2_to_halfraw2(kp[j], __NV_E4M3);
                    __half2_raw vh = __nv_cvt_fp8x2_to_halfraw2(vp[j], __NV_E4M3);
                    *reinterpret_cast<half2*>(&tile_K[tok_local][dim + 2 * j]) =
                        __hmul2(*reinterpret_cast<half2*>(&kh), ks2);
                    *reinterpret_cast<half2*>(&tile_V[tok_local][dim + 2 * j]) =
                        __hmul2(*reinterpret_cast<half2*>(&vh), vs2);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    tile_K[tok_local][dim + j] = __float2half(0.f);
                    tile_V[tok_local][dim + j] = __float2half(0.f);
                }
            }
        }
        __syncthreads();

        const __half* qh = tile_Q + warp_id * HEAD_DIM;
        const float s0 = (lane < tile_tokens)
                       ? qk_dot(qh, tile_K[lane], scale) : -FLT_MAX;
        const float s1 = (lane + WARP_SIZE < tile_tokens)
                       ? qk_dot(qh, tile_K[lane + WARP_SIZE], scale) : -FLT_MAX;

        const float tile_max = warp_reduce_max(fmaxf(s0, s1));
        const float new_max  = fmaxf(running_max, tile_max);
        const float rescale  = (running_max == -FLT_MAX) ? 0.f
                                                         : __expf(running_max - new_max);
        running_max = new_max;

        const float e0 = (s0 > -FLT_MAX) ? __expf(s0 - running_max) : 0.f;
        const float e1 = (s1 > -FLT_MAX) ? __expf(s1 - running_max) : 0.f;
        my_scores[lane] = e0;
        if (lane + WARP_SIZE < TILE_V4) my_scores[lane + WARP_SIZE] = e1;
        __syncwarp();

        running_sum = running_sum * rescale + warp_reduce_sum(e0 + e1);

        acc0 *= rescale; acc1 *= rescale; acc2 *= rescale; acc3 *= rescale;
        for (int tt = 0; tt < tile_tokens; tt++) {
            const float w = my_scores[tt];
            const half2 va = *reinterpret_cast<const half2*>(&tile_V[tt][2 * lane]);
            const half2 vb = *reinterpret_cast<const half2*>(&tile_V[tt][2 * lane + 64]);
            const float2 fa = __half22float2(va);
            const float2 fb = __half22float2(vb);
            acc0 += w * fa.x; acc1 += w * fa.y;
            acc2 += w * fb.x; acc3 += w * fb.y;
        }
    }

    float* out = O_partial + base_hs * D;
    out[2 * lane]          = acc0;
    out[2 * lane + 1]      = acc1;
    out[2 * lane + 64]     = acc2;
    out[2 * lane + 64 + 1] = acc3;
    if (lane == 0) {
        m_partial[base_hs] = running_max;
        l_partial[base_hs] = running_sum;
    }
}

// ─── v4 host launcher ────────────────────────────────────────────────────────
torch::Tensor launch_fused_gqa_v4(
    torch::Tensor Q,    // [B, Hq, D]      fp16
    torch::Tensor K,    // [B, N, Hkv, D]  fp16
    torch::Tensor V,    // [B, N, Hkv, D]  fp16
    double scale)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "Q/K/V must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat16 &&
                K.dtype() == torch::kFloat16 &&
                V.dtype() == torch::kFloat16, "all inputs must be fp16");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "all inputs must be contiguous");

    const int B   = Q.size(0);
    const int Hq  = Q.size(1);
    const int D   = Q.size(2);
    const int N   = K.size(1);
    const int Hkv = K.size(2);

    TORCH_CHECK(D == HEAD_DIM, "v4 requires head_dim=128, got ", D);
    TORCH_CHECK(Hq % Hkv == 0, "Hq=", Hq, " must be divisible by Hkv=", Hkv);
    const int G = Hq / Hkv;
    // One warp per query head of the group, so the softmax reduction stays
    // inside a warp. G>32 would exceed 1024 threads/block.
    TORCH_CHECK(G >= 1 && G <= 32, "v4 requires 1 <= Hq/Hkv <= 32, got ", G);
    TORCH_CHECK(K.size(3) == D && V.size(3) == D, "K/V head_dim must equal ", D);
    TORCH_CHECK(V.size(1) == N && V.size(2) == Hkv,
                "V shape must be [B, N, Hkv, D] matching K");

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                           Q.device().index());
    if (sm_count <= 0) sm_count = 132;

    // ── split count chosen to land on WHOLE WAVES ────────────────────────────
    // The grid is (Hkv, splits, B), so it contributes Hkv*B blocks per split.
    // The GPU runs sm_count * blocks_per_sm blocks at once; anything past a
    // multiple of that leaves a mostly-empty tail wave. At 64K the naive
    // 512-token target gave 8*125 = 1000 blocks against a 792-block capacity:
    // one full wave plus a wave only 26% occupied, ~63% effective utilisation.
    // Rounding the split count to a whole number of waves removes the tail.
    const int SPLIT_TOKENS = 512;      // preferred tokens per split
    const int MAX_SPLITS   = 512;

    const int smem_for_occ = static_cast<int>(
        (2 * TILE_V4 * SMEM_STRIDE + G * HEAD_DIM) * sizeof(__half)
        + G * TILE_V4 * sizeof(float));
    cudaFuncSetAttribute(gqa_decode_v4_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_for_occ);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, gqa_decode_v4_kernel, G * WARP_SIZE, smem_for_occ);
    if (blocks_per_sm <= 0) blocks_per_sm = 1;

    const int blocks_per_split = std::max(1, Hkv * B);
    const int capacity  = sm_count * blocks_per_sm;          // resident blocks
    const int per_wave  = std::max(1, capacity / blocks_per_split);
    const int want      = (N + SPLIT_TOKENS - 1) / SPLIT_TOKENS;
    const int waves     = std::max(1, (want + per_wave / 2) / per_wave);
    // Never split finer than one tile, or blocks do less work than a full tile.
    const int split_cap = std::max(1, N / TILE_V4);

    int num_splits = std::min(waves * per_wave, split_cap);
    num_splits = std::max(1, std::min(num_splits, MAX_SPLITS));
    int split_len = (N + num_splits - 1) / num_splits;
    num_splits    = (N + split_len - 1) / split_len;

    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto O_partial = torch::empty({B, Hq, num_splits, D}, f32);
    auto m_partial = torch::empty({B, Hq, num_splits},    f32);
    auto l_partial = torch::empty({B, Hq, num_splits},    f32);
    auto Out       = torch::empty({B, Hq, D}, Q.options());

    const int smem = smem_for_occ;   // K tile + V tile + Q rows + score scratch

    dim3 gridA(Hkv, num_splits, B);
    gqa_decode_v4_kernel<<<gridA, dim3(G * WARP_SIZE), smem>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        O_partial.data_ptr<float>(),
        m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(),
        B, N, Hq, Hkv, D, G, num_splits, split_len,
        static_cast<float>(scale));

    dim3 gridB(Hq, B);
    gqa_combine_kernel<<<gridB, dim3(D)>>>(          // unchanged from v3
        O_partial.data_ptr<float>(),
        m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(),
        reinterpret_cast<__half*>(Out.data_ptr<at::Half>()),
        B, Hq, D, num_splits);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "gqa_decode v4 launch failed: ", cudaGetErrorString(err));
    return Out;
}

// ─── v4-fp8 host launcher ────────────────────────────────────────────────────
torch::Tensor launch_fused_gqa_v4_fp8(
    torch::Tensor Q,        // [B, Hq, D]          fp16
    torch::Tensor Kq,       // [B, N, Hkv, D]      uint8 (e4m3)
    torch::Tensor Vq,       // [B, N, Hkv, D]      uint8 (e4m3)
    torch::Tensor k_scale,  // [B, num_pages, Hkv] fp32
    torch::Tensor v_scale,  // [B, num_pages, Hkv] fp32
    double scale)
{
    TORCH_CHECK(Q.is_cuda() && Kq.is_cuda() && Vq.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be fp16");
    TORCH_CHECK(Kq.dtype() == torch::kUInt8 && Vq.dtype() == torch::kUInt8,
                "Kq/Vq must be uint8 (e4m3 bits) from quantize_kv_fp8");
    TORCH_CHECK(k_scale.dtype() == torch::kFloat32 &&
                v_scale.dtype() == torch::kFloat32, "scales must be fp32");
    TORCH_CHECK(Q.is_contiguous() && Kq.is_contiguous() && Vq.is_contiguous() &&
                k_scale.is_contiguous() && v_scale.is_contiguous(),
                "all inputs must be contiguous");

    const int B   = Q.size(0);
    const int Hq  = Q.size(1);
    const int D   = Q.size(2);
    const int N   = Kq.size(1);
    const int Hkv = Kq.size(2);

    TORCH_CHECK(D == HEAD_DIM, "fp8 path requires head_dim=128, got ", D);
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");
    const int G = Hq / Hkv;
    TORCH_CHECK(G >= 1 && G <= 32, "requires 1 <= Hq/Hkv <= 32, got ", G);
    TORCH_CHECK(Vq.size(1) == N && Vq.size(2) == Hkv, "Vq must match Kq");

    const int num_pages = (N + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS;
    TORCH_CHECK(k_scale.size(1) == num_pages && v_scale.size(1) == num_pages,
                "scale pages (", k_scale.size(1), ") disagree with N=", N,
                " at ", KV_PAGE_TOKENS, " tokens/page (expected ", num_pages, ")");

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                           Q.device().index());
    if (sm_count <= 0) sm_count = 132;

    const int smem = static_cast<int>(
        (2 * TILE_V4 * SMEM_STRIDE + G * HEAD_DIM) * sizeof(__half)
        + G * TILE_V4 * sizeof(float));
    cudaFuncSetAttribute(gqa_decode_v4_fp8_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, gqa_decode_v4_fp8_kernel, G * WARP_SIZE, smem);
    if (blocks_per_sm <= 0) blocks_per_sm = 1;

    const int SPLIT_TOKENS = 512;
    const int MAX_SPLITS   = 512;
    const int blocks_per_split = std::max(1, Hkv * B);
    const int per_wave = std::max(1, (sm_count * blocks_per_sm) / blocks_per_split);
    const int want     = (N + SPLIT_TOKENS - 1) / SPLIT_TOKENS;
    const int waves    = std::max(1, (want + per_wave / 2) / per_wave);
    const int split_cap = std::max(1, N / TILE_V4);

    int num_splits = std::min(waves * per_wave, split_cap);
    num_splits = std::max(1, std::min(num_splits, MAX_SPLITS));
    int split_len = (N + num_splits - 1) / num_splits;
    // Round up so every tile is exactly one quantisation page -- the kernel
    // loads one scale per tile and would otherwise straddle two pages.
    split_len = ((split_len + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS) * KV_PAGE_TOKENS;
    num_splits = (N + split_len - 1) / split_len;

    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto O_partial = torch::empty({B, Hq, num_splits, D}, f32);
    auto m_partial = torch::empty({B, Hq, num_splits},    f32);
    auto l_partial = torch::empty({B, Hq, num_splits},    f32);
    auto Out       = torch::empty({B, Hq, D}, Q.options());

    dim3 gridA(Hkv, num_splits, B);
    gqa_decode_v4_fp8_kernel<<<gridA, dim3(G * WARP_SIZE), smem>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __nv_fp8_storage_t*>(Kq.data_ptr<uint8_t>()),
        reinterpret_cast<const __nv_fp8_storage_t*>(Vq.data_ptr<uint8_t>()),
        k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
        O_partial.data_ptr<float>(), m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(),
        B, N, Hq, Hkv, D, G, num_splits, split_len, num_pages,
        static_cast<float>(scale));

    dim3 gridB(Hq, B);
    gqa_combine_kernel<<<gridB, dim3(D)>>>(
        O_partial.data_ptr<float>(), m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(),
        reinterpret_cast<__half*>(Out.data_ptr<at::Half>()),
        B, Hq, D, num_splits);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "gqa_decode v4-fp8 launch failed: ", cudaGetErrorString(err));
    return Out;
}

// ─── occupancy introspection ─────────────────────────────────────────────────
// Nsight Compute counters are unavailable on this cluster (no sudo), so expose
// the launch-config facts the CUDA runtime will tell us for free: registers,
// shared memory, and blocks resident per SM. This is what actually decides
// latency hiding, and it needs no profiling permissions.
std::vector<int64_t> gqa_kernel_info(int64_t hq, int64_t hkv) {
    const int G = static_cast<int>(hq / hkv);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    cudaFuncAttributes a3{}, a4{};
    cudaFuncGetAttributes(&a3, gqa_decode_splitkv_kernel);
    cudaFuncGetAttributes(&a4, gqa_decode_v4_kernel);

    const int smem3 = 2 * TILE_SIZE * SMEM_STRIDE * sizeof(__half);
    const int smem4 = (2 * TILE_V4 * SMEM_STRIDE + G * HEAD_DIM) * sizeof(__half)
                    + G * TILE_V4 * sizeof(float);
    cudaFuncSetAttribute(gqa_decode_splitkv_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem3);
    cudaFuncSetAttribute(gqa_decode_v4_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem4);

    int blocks3 = 0, blocks4 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks3, gqa_decode_splitkv_kernel, BLOCK_THREADS, smem3);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks4, gqa_decode_v4_kernel, G * WARP_SIZE, smem4);

    return {static_cast<int64_t>(a3.numRegs), static_cast<int64_t>(smem3),
            static_cast<int64_t>(blocks3), static_cast<int64_t>(BLOCK_THREADS),
            static_cast<int64_t>(a4.numRegs), static_cast<int64_t>(smem4),
            static_cast<int64_t>(blocks4), static_cast<int64_t>(G * WARP_SIZE),
            static_cast<int64_t>(sm_count)};
}
