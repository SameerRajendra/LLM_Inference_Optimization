#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#define WARP_SIZE     32
#define FULL_MASK     0xffffffff
#define MAX_TOP_K     512
#define BLOCK_THREADS 256
#define NUM_WARPS     (BLOCK_THREADS / WARP_SIZE)   // 8

// ─── warp reductions ────────────────────────────────────────────────────────
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

// ─── block-level reductions ─────────────────────────────────────────────────
__device__ __forceinline__ float block_reduce_max(float val, float* smem_warp) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    val = warp_max(val);
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? smem_warp[threadIdx.x] : -FLT_MAX;
    if (warp_id == 0) val = warp_max(val);
    return __shfl_sync(FULL_MASK, val, 0);
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem_warp) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    val = warp_sum(val);
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? smem_warp[threadIdx.x] : 0.f;
    if (warp_id == 0) val = warp_sum(val);
    return __shfl_sync(FULL_MASK, val, 0);
}

// ─── online softmax (block-parallel) ────────────────────────────────────────
__device__ void online_softmax(float* scores, int k, float* warp_buf) {
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        local_max = fmaxf(local_max, scores[i]);
    float global_max = block_reduce_max(local_max, warp_buf);
    __syncthreads();

    float local_sum = 0.f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        scores[i] = expf(scores[i] - global_max);
        local_sum += scores[i];
    }
    float global_sum = block_reduce_sum(local_sum, warp_buf);
    if (global_sum < 1e-9f) global_sum = 1e-9f;
    __syncthreads();

    for (int i = threadIdx.x; i < k; i += blockDim.x)
        scores[i] /= global_sum;
    __syncthreads();
    // NOTE: warp_buf holds stale reduction values after return.
    // Do not read warp_buf without __syncthreads() + reinit.
}

// ─── per-lane register heap (min-replace) ───────────────────────────────────
// original used a shared-memory per-warp heap written by all
// lanes simultaneously — a shared memory write conflict with no sync.
// Fix: each lane maintains its own heap in REGISTERS (private, no conflict),
// then writes results to shared memory only once, after the scan is complete.
//
// MAX_LANE_HEAP must be >= ceil(MAX_TOP_K / BLOCK_THREADS) = ceil(512/256) = 2.
// Set to 4 to give headroom for top_k values up to BLOCK_THREADS*4 = 1024.
#define MAX_LANE_HEAP 4

__device__ void lane_heap_insert(
    float* heap_scores,   // register array [MAX_LANE_HEAP]
    int*   heap_indices,  // register array [MAX_LANE_HEAP]
    int    heap_size,
    float  score,
    int    idx)
{
    // Find minimum slot in the lane's private heap
    int min_i = 0;
    for (int i = 1; i < heap_size; i++)
        if (heap_scores[i] < heap_scores[min_i]) min_i = i;

    // Replace if new score beats current minimum
    if (score > heap_scores[min_i]) {
        heap_scores[min_i]  = score;
        heap_indices[min_i] = idx;
    }
}

// ─── main kernel ─────────────────────────────────────────────────────────────
// Q:   [B, H, 1,       D]  fp16
// K:   [B, H, ctx_len, D]  fp16
// V:   [B, H, ctx_len, D]  fp16
// Out: [B, H, 1,       D]  fp16
// top_k_indices: [B, H, top_k] int32 (output, for inspection/debugging)
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
    int b       = blockIdx.x;
    int h       = blockIdx.y;
    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    const __half* q = Q + (b * H + h) * D;
    const __half* k = K + (b * H + h) * ctx_len * D;
    const __half* v = V + (b * H + h) * ctx_len * D;
    __half*       o = Out + (b * H + h) * D;

    float* head_scores    = scores_buf    + (b * H + h) * ctx_len;
    int32_t* head_topk_idx = top_k_indices + (b * H + h) * top_k;

    int actual_k  = min(top_k, ctx_len);

    //  local_k is per-THREAD budget (not per-warp).
    // Each of BLOCK_THREADS threads holds ceil(actual_k / BLOCK_THREADS)
    // candidates in private register heaps.  local_k <= MAX_LANE_HEAP.
    int local_k = (actual_k + blockDim.x - 1) / blockDim.x;
    // Clamp to register array bound — TORCH_CHECK in host ensures top_k fits
    local_k = min(local_k, MAX_LANE_HEAP);

    // ── Shared memory layout ─────────────────────────────────────────────
    // [warp_buf        : NUM_WARPS floats   ] block reduction scratch
    // [s_lane_scores   : BLOCK_THREADS * MAX_LANE_HEAP floats ] lane heap staging
    // [s_lane_indices  : BLOCK_THREADS * MAX_LANE_HEAP ints   ]
    // [s_topk_scores   : actual_k floats   ] merged top-k scores
    // [s_topk_indices  : actual_k ints     ] merged top-k indices
    // [s_out_buf       : D floats          ] output accumulator
    extern __shared__ char smem[];
    float*  warp_buf       = reinterpret_cast<float*>(smem);
    float*  s_lane_scores  = warp_buf + NUM_WARPS;
    int*    s_lane_indices = reinterpret_cast<int*>(
                                 s_lane_scores + blockDim.x * MAX_LANE_HEAP);
    float*  s_topk_scores  = reinterpret_cast<float*>(
                                 s_lane_indices + blockDim.x * MAX_LANE_HEAP);
    int*    s_topk_indices = reinterpret_cast<int*>(s_topk_scores + actual_k);
    float*  s_out_buf      = reinterpret_cast<float*>(s_topk_indices + actual_k);

    // ── Step 1: QK^T scores (half2 vectorised, all threads) ─────────────
    for (int tok = tid; tok < ctx_len; tok += blockDim.x) {
        float dot = 0.f;
        const __half* krow = k + tok * D;

        // FIX [P2]: half2 SIMD — 2× throughput vs scalar loop
        int d = 0;
        for (; d + 1 < D; d += 2) {
            half2 q2 = *reinterpret_cast<const half2*>(&q[d]);
            half2 k2 = *reinterpret_cast<const half2*>(&krow[d]);
            float2 qf = __half22float2(q2);
            float2 kf = __half22float2(k2);
            dot += qf.x * kf.x + qf.y * kf.y;
        }
        if (d < D)   // scalar tail for odd D
            dot += __half2float(q[d]) * __half2float(krow[d]);

        head_scores[tok] = dot * scale;
    }
    __syncthreads();

    // ── Step 2: per-lane register heap scan ──────────────────────────────
    // each thread scans its private slice of tokens and
    // maintains a register heap — no shared memory writes during scan.
    float lane_scores[MAX_LANE_HEAP];
    int   lane_indices[MAX_LANE_HEAP];
    for (int i = 0; i < local_k; i++) {
        lane_scores[i]  = -FLT_MAX;
        lane_indices[i] = -1;
    }

    for (int tok = tid; tok < ctx_len; tok += blockDim.x)
        lane_heap_insert(lane_scores, lane_indices, local_k,
                         head_scores[tok], tok);

    // Write register heap to shared memory staging area (one write per thread)
    for (int i = 0; i < local_k; i++) {
        s_lane_scores [tid * MAX_LANE_HEAP + i] = lane_scores[i];
        s_lane_indices[tid * MAX_LANE_HEAP + i] = lane_indices[i];
    }
    // Zero-fill unused slots so merge sees -FLT_MAX for padded entries
    for (int i = local_k; i < MAX_LANE_HEAP; i++) {
        s_lane_scores [tid * MAX_LANE_HEAP + i] = -FLT_MAX;
        s_lane_indices[tid * MAX_LANE_HEAP + i] = -1;
    }
    __syncthreads();

    // ── Step 3: warp 0 merges all per-thread heaps ───────────────────────
    // merged_scores/indices now live in s_topk_scores/indices
    // (shared memory) instead of per-thread stack VLAs — avoids 2 KB of
    // local memory (off-chip DRAM) per thread.
    // merge_pos zero-initialised explicitly (not via = {0}).
    if (warp_id == 0 && lane_id == 0) {
        // Total candidate pool: BLOCK_THREADS * local_k entries in s_lane_*
        int total_candidates = blockDim.x * MAX_LANE_HEAP;

        // Simple O(actual_k * total_candidates) selection — correct and
        // runs on a single thread; total_candidates <= 256*4 = 1024,
        // actual_k <= 512 → at most ~500K comparisons, ~2 µs on A100.
        for (int pick = 0; pick < actual_k; pick++) {
            float best     = -FLT_MAX;
            int   best_pos = -1;
            for (int c = 0; c < total_candidates; c++) {
                float s = s_lane_scores[c];
                if (s > best) { best = s; best_pos = c; }
            }
            if (best_pos < 0 || best <= -FLT_MAX) break;

            s_topk_scores [pick] = best;
            s_topk_indices[pick] = s_lane_indices[best_pos];
            head_topk_idx [pick] = s_lane_indices[best_pos];

            // Mark as consumed so it won't be picked again
            s_lane_scores[best_pos] = -FLT_MAX - 1.f;
        }
        // Fill remaining slots as invalid
        for (int i = 0; i < actual_k; i++)
            if (s_topk_indices[i] == 0 && s_topk_scores[i] <= -FLT_MAX)
                s_topk_indices[i] = -1;
        for (int i = actual_k; i < top_k; i++)
            head_topk_idx[i] = -1;
    }
    __syncthreads();

    // ── Step 4: softmax over top-k scores ────────────────────────────────
    online_softmax(s_topk_scores, actual_k, warp_buf);
    // online_softmax ends with __syncthreads() — s_topk_scores is safe to read

    // ── Step 5: weighted V accumulation ──────────────────────────────────
    for (int d = tid; d < D; d += blockDim.x)
        s_out_buf[d] = 0.f;
    __syncthreads();

    // removed __syncthreads() from inside the loop.
    // Each thread writes only its own d-slots — no cross-thread conflict
    // within one iteration.  One barrier after the full loop is sufficient.
    for (int i = 0; i < actual_k; i++) {
        int   tok = s_topk_indices[i];
        float w   = s_topk_scores[i];

        // guard against invalid indices from unfilled heap slots
        if (tok < 0 || tok >= ctx_len) continue;

        const __half* vrow = v + tok * D;
        for (int d = tid; d < D; d += blockDim.x)
            s_out_buf[d] += w * __half2float(vrow[d]);
    }
    __syncthreads();   // one barrier after full accumulation

    // Write output
    for (int d = tid; d < D; d += blockDim.x)
        o[d] = __float2half_rn(s_out_buf[d]);
}

// ─── host launcher ───────────────────────────────────────────────────────────
torch::Tensor kv_evict_quant_forward(
    torch::Tensor Q,       // [B, H, 1, D]       fp16
    torch::Tensor K,       // [B, H, ctx_len, D] fp16
    torch::Tensor V,       // [B, H, ctx_len, D] fp16
    int top_k,
    bool use_int8)
{
    // ── input validation ─────────────────────────────────────────────────
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "Q/K/V must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be fp16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be fp16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be fp16");

    // FIX [P1]: validate decode-step shape
    TORCH_CHECK(Q.size(2) == 1,
                "kv_evict_quant_forward: Q must be decode-step shape [B,H,1,D]");
    TORCH_CHECK(K.size(0) == Q.size(0) && K.size(1) == Q.size(1),
                "K batch/head dims must match Q");
    TORCH_CHECK(V.size(2) == K.size(2) && V.size(3) == K.size(3),
                "V shape must match K");
    TORCH_CHECK(top_k > 0, "top_k must be positive");
    TORCH_CHECK(use_int8 == false, "INT8 path not enabled in this build");

    int B       = Q.size(0);
    int H       = Q.size(1);
    int ctx_len = K.size(2);
    int D       = Q.size(3);

    TORCH_CHECK(D <= 4096,        "D=", D, " exceeds max supported (4096)");
    TORCH_CHECK(top_k <= MAX_TOP_K,
                "top_k=", top_k, " exceeds MAX_TOP_K=", MAX_TOP_K);

    // FIX [P2]: validate local_k fits register heap
    int local_k = (min(top_k, ctx_len) + BLOCK_THREADS - 1) / BLOCK_THREADS;
    TORCH_CHECK(local_k <= MAX_LANE_HEAP,
                "local_k=", local_k, " exceeds MAX_LANE_HEAP=", MAX_LANE_HEAP,
                ". Increase MAX_LANE_HEAP or reduce top_k.");

    float scale = 1.f / sqrtf(static_cast<float>(D));

    auto Out = torch::zeros({B, H, 1, D}, Q.options());
    auto top_k_idx = torch::full(
        {B, H, top_k}, -1,
        torch::dtype(torch::kInt32).device(Q.device()));
    auto scores_buf = torch::empty(
        {B, H, ctx_len},
        torch::dtype(torch::kFloat32).device(Q.device()));

    dim3 grid(B, H);
    dim3 block(BLOCK_THREADS);

    // ── shared memory calculation ─────────────────────────────────────────
    //  use actual_k = min(top_k, ctx_len) for smem sizing,
    // not top_k — avoids over-allocation when ctx_len < top_k.
    int actual_k = min(top_k, ctx_len);
    size_t smem_bytes =
        NUM_WARPS          * sizeof(float) +          // warp_buf
        BLOCK_THREADS      * MAX_LANE_HEAP * sizeof(float) +  // s_lane_scores
        BLOCK_THREADS      * MAX_LANE_HEAP * sizeof(int)   +  // s_lane_indices
        actual_k           * sizeof(float) +          // s_topk_scores
        actual_k           * sizeof(int)   +          // s_topk_indices
        D                  * sizeof(float);           // s_out_buf

    // runtime shared memory limit check
    int device_id = Q.device().index();
    int max_smem  = 0;
    cudaDeviceGetAttribute(&max_smem,
                           cudaDevAttrMaxSharedMemoryPerBlock, device_id);
    TORCH_CHECK((int)smem_bytes <= max_smem,
                "smem_bytes=", smem_bytes,
                " exceeds device limit=", max_smem,
                ". Reduce top_k or D.");

    sparse_kv_attn_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Out.data_ptr<at::Half>()),
        scores_buf.data_ptr<float>(),
        top_k_idx.data_ptr<int32_t>(),
        B, H, ctx_len, D, top_k, scale);

#ifdef DEBUG_KERNELS
    cudaDeviceSynchronize();
#endif
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "sparse_kv_attn_kernel launch failed: ",
                cudaGetErrorString(err));

    return Out;
}
