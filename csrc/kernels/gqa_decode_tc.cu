/*
 * gqa_decode_tc.cu — Tensor-Core + cp.async double-buffered GQA decode kernel (WP-3)
 *
 * Solves the v4 scalar-FMA instruction-issue bottleneck by computing QK^T and PV
 * with tensor-core mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 while
 * overlapping global->shared loads via an explicit 2-stage cp.async pipeline.
 *
 * Layouts:
 *   Q         [B, Hq, D] or [B, S_q, Hq, D]   fp16
 *   K, V      [B, N, Hkv, D]                  fp16 (FP8=false) or uint8 e4m3 (FP8=true)
 *   k_scale,
 *   v_scale   [B, num_pages, Hkv]             fp32 (FP8=true only; KV_PAGE_TOKENS=64)
 *   Out       same rank/shape as Q            fp16
 *
 * Design:
 *   - M=16 of mma.sync packs up to 16 (s, g) rows: G query heads of a KV group
 *     times S_q speculative-decode query tokens. When G*S_q > 16, grid.x loops
 *     over ceil(G*S_q / 16) m-tiles so any (G, S_q) works with full K/V reuse
 *     inside each m-tile.
 *   - 4 warps (128 threads) per block split BLOCK_N=64 tokens (16 tokens/warp =
 *     two m16n8k16 N-tiles per warp). Each warp keeps its P fragment in registers
 *     for P @ V directly (no P smem round-trip; exact mma D->A register match).
 *   - FP8 mode streams raw e4m3 bytes via cp.async (32 KB smem @ D=128 vs 64 KB)
 *     and applies k_scale / v_scale in fp32 outside mma.sync (zero fp16 scale
 *     underflow/overflow).
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

namespace {

constexpr int TC_BLOCK_N = 64;
constexpr int TC_WARPS   = 4;
constexpr int TC_THREADS = TC_WARPS * 32;  // 128
constexpr uint32_t TC_FULL_MASK = 0xffffffffu;

static_assert(KV_PAGE_TOKENS % TC_BLOCK_N == 0,
              "TC_BLOCK_N must divide KV_PAGE_TOKENS evenly for single-page FP8 scale lookup");

// ─── inline PTX primitives (sm_80+: Ada sm_89 and Hopper sm_90a) ────────────

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src, int src_size) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint64_t src = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(gmem_src));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(dst), "l"(src), "r"(src_size)
        : "memory");
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int N_WAIT>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT) : "memory");
#endif
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
#else
    r0 = r1 = r2 = r3 = 0;
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
#else
    r0 = r1 = r2 = r3 = 0;
#endif
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ uint32_t pack_half2_rn(float lo, float hi) {
    __half2 h2 = __float22half2_rn(make_float2(lo, hi));
    __half2_raw raw = h2;
    return static_cast<uint32_t>(raw.x) | (static_cast<uint32_t>(raw.y) << 16);
}

__device__ __forceinline__ uint32_t fp8x2_to_u32_f16x2(uint16_t raw_pair) {
    __half2_raw kh = __nv_cvt_fp8x2_to_halfraw2(raw_pair, __NV_E4M3);
    return static_cast<uint32_t>(kh.x) | (static_cast<uint32_t>(kh.y) << 16);
}

// 4-lane reduction across tid_in_group = lane & 3 (only masks 1 and 2).
__device__ __forceinline__ float quad_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(TC_FULL_MASK, v, 1));
    v = fmaxf(v, __shfl_xor_sync(TC_FULL_MASK, v, 2));
    return v;
}

__device__ __forceinline__ float quad_reduce_sum(float v) {
    v += __shfl_xor_sync(TC_FULL_MASK, v, 1);
    v += __shfl_xor_sync(TC_FULL_MASK, v, 2);
    return v;
}

// ─── Pass 1: Tensor-Core split-KV decode kernel ──────────────────────────────

template <int HEAD_DIM, int BLOCK_N, bool FP8>
__global__ void __launch_bounds__(TC_THREADS, 2) gqa_decode_tc_kernel(
    const __half* __restrict__ Q,          // [B, S_q, Hq, HEAD_DIM]
    const void*   __restrict__ K_ptr,      // [B, N, Hkv, HEAD_DIM] fp16 or uint8
    const void*   __restrict__ V_ptr,      // [B, N, Hkv, HEAD_DIM] fp16 or uint8
    const float*  __restrict__ k_scale,    // [B, num_pages, Hkv] (unused if !FP8)
    const float*  __restrict__ v_scale,    // [B, num_pages, Hkv] (unused if !FP8)
    float*        __restrict__ O_partial,  // [B * S_q, Hq, num_splits, HEAD_DIM]
    float*        __restrict__ m_partial,  // [B * S_q, Hq, num_splits]
    float*        __restrict__ l_partial,  // [B * S_q, Hq, num_splits]
    int B, int S_q, int N, int Hq, int Hkv, int G,
    int num_m_tiles, int num_splits, int split_len, int num_pages,
    float scale, int causal)
{
    constexpr int ELEM_BYTES      = FP8 ? 1 : 2;
    constexpr int ELEMS_PER_CHUNK = 16 / ELEM_BYTES;               // 8 (fp16) or 16 (fp8)
    constexpr int CHUNKS_PER_ROW  = HEAD_DIM / ELEMS_PER_CHUNK;
    constexpr int SWIZ_MASK       = (CHUNKS_PER_ROW >= 8) ? 7 : (CHUNKS_PER_ROW - 1);
    constexpr int TOTAL_CHUNKS    = BLOCK_N * CHUNKS_PER_ROW;
    constexpr int STAGE_BYTES     = BLOCK_N * HEAD_DIM * ELEM_BYTES;
    constexpr int K_SMEM_BYTES    = 2 * STAGE_BYTES;
    constexpr int KV_SMEM_BYTES   = 2 * K_SMEM_BYTES;              // double-buffered K & V
    constexpr int Q_CHUNKS        = HEAD_DIM / 8;
    constexpr int Q_SWIZ_MASK     = (Q_CHUNKS >= 8) ? 7 : (Q_CHUNKS - 1);
    constexpr int NUM_K_STEPS     = HEAD_DIM / 16;                 // 4 or 8
    constexpr int NUM_N8_TILES    = HEAD_DIM / 8;                  // 8 or 16

    const int kv_m    = blockIdx.x;
    const int kv_head = kv_m / num_m_tiles;
    const int m_tile  = kv_m % num_m_tiles;
    const int split   = blockIdx.y;
    const int b       = blockIdx.z;

    const int tid          = threadIdx.x;                          // 0..127
    const int warp_id      = tid >> 5;                             // 0..3
    const int lane         = tid & 31;                             // 0..31
    const int group_id     = lane >> 2;                            // 0..7
    const int tid_in_group = lane & 3;                             // 0..3

    const int m_base  = m_tile * 16;
    const int m_valid = min(16, G * S_q - m_base);

    const int split_start = split * split_len;
    const int split_end   = min(split_start + split_len, N);

    if (split_start >= split_end) {
        for (int i = tid; i < m_valid * HEAD_DIM; i += TC_THREADS) {
            const int m       = i / HEAD_DIM;
            const int d       = i % HEAD_DIM;
            const int row_idx = m_base + m;
            const int s       = row_idx / G;
            const int g       = row_idx % G;
            const int64_t hs  = (static_cast<int64_t>(b * S_q + s) * Hq + (kv_head * G + g))
                              * num_splits + split;
            O_partial[hs * HEAD_DIM + d] = 0.f;
        }
        if (tid < m_valid) {
            const int row_idx = m_base + tid;
            const int s       = row_idx / G;
            const int g       = row_idx % G;
            const int64_t hs  = (static_cast<int64_t>(b * S_q + s) * Hq + (kv_head * G + g))
                              * num_splits + split;
            m_partial[hs] = -FLT_MAX;
            l_partial[hs] = 0.f;
        }
        return;
    }

    // Dynamic shared memory layout (16B-aligned base):
    //   [0, KV_SMEM_BYTES) : double-buffered K (2 stages) then V (2 stages)
    //   tile_Q             : [16][HEAD_DIM] fp16 (swizzled, 16B aligned)
    //   smem_red           : [2][TC_WARPS][16] float
    //   smem_rmax          : [16] float
    //   smem_rsum          : [16] float
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem_k   = smem_raw;
    uint8_t* smem_v   = smem_raw + K_SMEM_BYTES;
    __half*  tile_Q   = reinterpret_cast<__half*>(smem_raw + KV_SMEM_BYTES);
    float*   smem_red = reinterpret_cast<float*>(tile_Q + 16 * HEAD_DIM);
    float*   smem_rmax = smem_red + 2 * TC_WARPS * 16;
    float*   smem_rsum = smem_rmax + 16;

    // ── Load Q into swizzled smem, then register fragments q_frag[NUM_K_STEPS][4] ──
    for (int i = tid; i < 16 * Q_CHUNKS; i += TC_THREADS) {
        const int m       = i / Q_CHUNKS;
        const int c8      = i % Q_CHUNKS;
        const int c8_swiz = c8 ^ (m & Q_SWIZ_MASK);
        uint4* dst = reinterpret_cast<uint4*>(tile_Q + m * HEAD_DIM + c8_swiz * 8);
        const int row_idx = m_base + m;
        if (row_idx < G * S_q) {
            const int s       = row_idx / G;
            const int q_head  = kv_head * G + (row_idx % G);
            const int64_t qoff = (static_cast<int64_t>(b * S_q + s) * Hq + q_head) * HEAD_DIM
                               + c8 * 8;
            *dst = *reinterpret_cast<const uint4*>(Q + qoff);
        } else {
            *dst = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    uint32_t q_frag[NUM_K_STEPS][4];
    {
        const int q_mat  = lane >> 3;           // 0..3
        const int m_ld   = ((q_mat & 1) << 3) + (lane & 7);
        const int kh_off = (q_mat >> 1);        // 0 or 1
        #pragma unroll
        for (int ks = 0; ks < NUM_K_STEPS; ks++) {
            const int c8      = 2 * ks + kh_off;
            const int c8_swiz = c8 ^ (m_ld & Q_SWIZ_MASK);
            ldmatrix_x4(q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3],
                        tile_Q + m_ld * HEAD_DIM + c8_swiz * 8);
        }
    }

    // Per-row causal upper limit (exclusive token index in [0, N]).
    const int row_idx_lo = m_base + group_id;
    const int row_idx_hi = m_base + group_id + 8;
    const int causal_end_lo =
        (causal && row_idx_lo < G * S_q) ? (N - S_q + (row_idx_lo / G) + 1) : N;
    const int causal_end_hi =
        (causal && row_idx_hi < G * S_q) ? (N - S_q + (row_idx_hi / G) + 1) : N;

    const uint8_t* K_bytes = reinterpret_cast<const uint8_t*>(K_ptr);
    const uint8_t* V_bytes = reinterpret_cast<const uint8_t*>(V_ptr);

    auto issue_tile_cp_async = [&](int tile_idx, int stage) {
        const int t_start = split_start + tile_idx * BLOCK_N;
        const int t_toks  = min(BLOCK_N, split_end - t_start);
        uint8_t* kd = smem_k + stage * STAGE_BYTES;
        uint8_t* vd = smem_v + stage * STAGE_BYTES;
        #pragma unroll
        for (int idx = tid; idx < TOTAL_CHUNKS; idx += TC_THREADS) {
            const int tok_local = idx / CHUNKS_PER_ROW;
            const int c         = idx % CHUNKS_PER_ROW;
            const int c_swiz    = c ^ (tok_local & SWIZ_MASK);
            const int smem_off  = (tok_local * CHUNKS_PER_ROW + c_swiz) * 16;
            const bool valid    = (tok_local < t_toks);
            const int src_size  = valid ? 16 : 0;
            const int64_t goff  =
                ((static_cast<int64_t>(b) * N + (t_start + tok_local)) * Hkv + kv_head)
                * (HEAD_DIM * ELEM_BYTES) + c * 16;
            cp_async_16(kd + smem_off, valid ? (K_bytes + goff) : K_bytes, src_size);
            cp_async_16(vd + smem_off, valid ? (V_bytes + goff) : V_bytes, src_size);
        }
        cp_async_commit();
    };

    const int num_tiles = (split_end - split_start + BLOCK_N - 1) / BLOCK_N;

    // Prologue: prefetch tile 0 (and tile 1 if present).
    issue_tile_cp_async(0, 0);
    if (num_tiles > 1) {
        issue_tile_cp_async(1, 1);
    }

    float rmax_lo = -FLT_MAX, rmax_hi = -FLT_MAX;
    float rsum_lo = 0.f,      rsum_hi = 0.f;
    float o_acc[NUM_N8_TILES][4];
    #pragma unroll
    for (int nt = 0; nt < NUM_N8_TILES; nt++) {
        o_acc[nt][0] = 0.f;
        o_acc[nt][1] = 0.f;
        o_acc[nt][2] = 0.f;
        o_acc[nt][3] = 0.f;
    }

    const int warp_tok_base = warp_id * 16;

    for (int t = 0; t < num_tiles; t++) {
        const int stage       = t & 1;
        const int tile_start  = split_start + t * BLOCK_N;
        const int tile_tokens = min(BLOCK_N, split_end - tile_start);

        if (t + 1 < num_tiles) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        const uint8_t* k_tile = smem_k + stage * STAGE_BYTES;
        const uint8_t* v_tile = smem_v + stage * STAGE_BYTES;

        // ── 1. QK^T via mma.sync (two m16n8k16 N-tiles for this warp's 16 tokens) ──
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        float s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;

        if constexpr (!FP8) {
            const __half* kt_h = reinterpret_cast<const __half*>(k_tile);
            const int ld_mat = lane >> 3;        // 0..3
            const int tok_ld = warp_tok_base + ((ld_mat >> 1) << 3) + (lane & 7);
            const int kh_off = ld_mat & 1;       // 0 or 1
            #pragma unroll
            for (int ks = 0; ks < NUM_K_STEPS; ks++) {
                const int c8      = 2 * ks + kh_off;
                const int c8_swiz = c8 ^ (tok_ld & SWIZ_MASK);
                uint32_t b0_n0, b1_n0, b0_n1, b1_n1;
                ldmatrix_x4(b0_n0, b1_n0, b0_n1, b1_n1,
                            kt_h + tok_ld * HEAD_DIM + c8_swiz * 8);
                mma_m16n8k16(s0, s1, s2, s3,
                             q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3],
                             b0_n0, b1_n0);
                mma_m16n8k16(s4, s5, s6, s7,
                             q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3],
                             b0_n1, b1_n1);
            }
        } else {
            const int tok0 = warp_tok_base + group_id;
            const int tok1 = tok0 + 8;
            const int d_off0 = 2 * tid_in_group;
            const int d_off1 = d_off0 + 8;
            #pragma unroll
            for (int ks = 0; ks < NUM_K_STEPS; ks++) {
                const int sw0 = (ks ^ (tok0 & SWIZ_MASK)) * 16;
                const int sw1 = (ks ^ (tok1 & SWIZ_MASK)) * 16;
                const uint16_t raw_b0_n0 = *reinterpret_cast<const uint16_t*>(
                    k_tile + tok0 * HEAD_DIM + sw0 + d_off0);
                const uint16_t raw_b1_n0 = *reinterpret_cast<const uint16_t*>(
                    k_tile + tok0 * HEAD_DIM + sw0 + d_off1);
                const uint16_t raw_b0_n1 = *reinterpret_cast<const uint16_t*>(
                    k_tile + tok1 * HEAD_DIM + sw1 + d_off0);
                const uint16_t raw_b1_n1 = *reinterpret_cast<const uint16_t*>(
                    k_tile + tok1 * HEAD_DIM + sw1 + d_off1);
                mma_m16n8k16(s0, s1, s2, s3,
                             q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3],
                             fp8x2_to_u32_f16x2(raw_b0_n0), fp8x2_to_u32_f16x2(raw_b1_n0));
                mma_m16n8k16(s4, s5, s6, s7,
                             q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3],
                             fp8x2_to_u32_f16x2(raw_b0_n1), fp8x2_to_u32_f16x2(raw_b1_n1));
            }
        }

        const int page = FP8 ? (tile_start / KV_PAGE_TOKENS) : 0;
        const float ks_val = FP8 ? k_scale[(b * num_pages + page) * Hkv + kv_head] : 1.f;
        const float vs_val = FP8 ? v_scale[(b * num_pages + page) * Hkv + kv_head] : 1.f;
        const float eff_scale = scale * ks_val;

        const int end_lo = min(tile_tokens, max(0, causal_end_lo - tile_start));
        const int end_hi = min(tile_tokens, max(0, causal_end_hi - tile_start));

        const int col0 = warp_tok_base + 2 * tid_in_group;
        const int col1 = col0 + 1;
        const int col4 = col0 + 8;
        const int col5 = col0 + 9;

        s0 = (col0 < end_lo) ? (s0 * eff_scale) : -FLT_MAX;
        s1 = (col1 < end_lo) ? (s1 * eff_scale) : -FLT_MAX;
        s4 = (col4 < end_lo) ? (s4 * eff_scale) : -FLT_MAX;
        s5 = (col5 < end_lo) ? (s5 * eff_scale) : -FLT_MAX;

        s2 = (col0 < end_hi) ? (s2 * eff_scale) : -FLT_MAX;
        s3 = (col1 < end_hi) ? (s3 * eff_scale) : -FLT_MAX;
        s6 = (col4 < end_hi) ? (s6 * eff_scale) : -FLT_MAX;
        s7 = (col5 < end_hi) ? (s7 * eff_scale) : -FLT_MAX;

        // ── 2. Online softmax across the 4 warps ──
        const float wmax_lo = quad_reduce_max(fmaxf(fmaxf(s0, s1), fmaxf(s4, s5)));
        const float wmax_hi = quad_reduce_max(fmaxf(fmaxf(s2, s3), fmaxf(s6, s7)));

        float* red_stage = smem_red + stage * (TC_WARPS * 16);
        if (tid_in_group == 0) {
            red_stage[warp_id * 16 + group_id]     = wmax_lo;
            red_stage[warp_id * 16 + group_id + 8] = wmax_hi;
        }
        __syncthreads();

        const float tmax_lo = fmaxf(
            fmaxf(red_stage[0 * 16 + group_id], red_stage[1 * 16 + group_id]),
            fmaxf(red_stage[2 * 16 + group_id], red_stage[3 * 16 + group_id]));
        const float tmax_hi = fmaxf(
            fmaxf(red_stage[0 * 16 + group_id + 8], red_stage[1 * 16 + group_id + 8]),
            fmaxf(red_stage[2 * 16 + group_id + 8], red_stage[3 * 16 + group_id + 8]));

        const float new_max_lo = fmaxf(rmax_lo, tmax_lo);
        const float new_max_hi = fmaxf(rmax_hi, tmax_hi);
        const float rescale_lo = (rmax_lo == -FLT_MAX) ? 0.f : __expf(rmax_lo - new_max_lo);
        const float rescale_hi = (rmax_hi == -FLT_MAX) ? 0.f : __expf(rmax_hi - new_max_hi);
        rmax_lo = new_max_lo;
        rmax_hi = new_max_hi;

        const float p0 = (s0 > -FLT_MAX) ? __expf(s0 - rmax_lo) : 0.f;
        const float p1 = (s1 > -FLT_MAX) ? __expf(s1 - rmax_lo) : 0.f;
        const float p4 = (s4 > -FLT_MAX) ? __expf(s4 - rmax_lo) : 0.f;
        const float p5 = (s5 > -FLT_MAX) ? __expf(s5 - rmax_lo) : 0.f;

        const float p2 = (s2 > -FLT_MAX) ? __expf(s2 - rmax_hi) : 0.f;
        const float p3 = (s3 > -FLT_MAX) ? __expf(s3 - rmax_hi) : 0.f;
        const float p6 = (s6 > -FLT_MAX) ? __expf(s6 - rmax_hi) : 0.f;
        const float p7 = (s7 > -FLT_MAX) ? __expf(s7 - rmax_hi) : 0.f;

        rsum_lo = rsum_lo * rescale_lo + quad_reduce_sum((p0 + p1) + (p4 + p5));
        rsum_hi = rsum_hi * rescale_hi + quad_reduce_sum((p2 + p3) + (p6 + p7));

        // ── 3. Rescale previous O_acc and accumulate P @ V (register-to-register A) ──
        #pragma unroll
        for (int nt = 0; nt < NUM_N8_TILES; nt++) {
            o_acc[nt][0] *= rescale_lo;
            o_acc[nt][1] *= rescale_lo;
            o_acc[nt][2] *= rescale_hi;
            o_acc[nt][3] *= rescale_hi;
        }

        const uint32_t a0 = pack_half2_rn(p0, p1);
        const uint32_t a1 = pack_half2_rn(p2, p3);
        const uint32_t a2 = pack_half2_rn(p4, p5);
        const uint32_t a3 = pack_half2_rn(p6, p7);

        if constexpr (!FP8) {
            const __half* vt_h = reinterpret_cast<const __half*>(v_tile);
            const int ld_mat = lane >> 3;        // 0..3
            const int tok_ld = warp_tok_base + ((ld_mat & 1) << 3) + (lane & 7);
            const int n_off  = ld_mat >> 1;      // 0 or 1
            #pragma unroll
            for (int np = 0; np < NUM_N8_TILES / 2; np++) {
                const int c8      = 2 * np + n_off;
                const int c8_swiz = c8 ^ (tok_ld & SWIZ_MASK);
                uint32_t b0_0, b1_0, b0_1, b1_1;
                ldmatrix_x4_trans(b0_0, b1_0, b0_1, b1_1,
                                  vt_h + tok_ld * HEAD_DIM + c8_swiz * 8);
                mma_m16n8k16(o_acc[2 * np][0], o_acc[2 * np][1],
                             o_acc[2 * np][2], o_acc[2 * np][3],
                             a0, a1, a2, a3, b0_0, b1_0);
                mma_m16n8k16(o_acc[2 * np + 1][0], o_acc[2 * np + 1][1],
                             o_acc[2 * np + 1][2], o_acc[2 * np + 1][3],
                             a0, a1, a2, a3, b0_1, b1_1);
            }
        } else {
            const int tok0 = warp_tok_base + 2 * tid_in_group;
            const int tok1 = tok0 + 1;
            const int tok8 = tok0 + 8;
            const int tok9 = tok0 + 9;
            #pragma unroll
            for (int nt = 0; nt < NUM_N8_TILES; nt++) {
                const int c16   = nt >> 1;
                const int d_in  = ((nt & 1) << 3) + group_id;
                const int sw0   = (c16 ^ (tok0 & SWIZ_MASK)) * 16 + d_in;
                const int sw1   = (c16 ^ (tok1 & SWIZ_MASK)) * 16 + d_in;
                const uint16_t raw0 =
                    static_cast<uint16_t>(v_tile[tok0 * HEAD_DIM + sw0]) |
                    (static_cast<uint16_t>(v_tile[tok1 * HEAD_DIM + sw1]) << 8);
                // (tok0 + 8) & SWIZ_MASK == tok0 & SWIZ_MASK since SWIZ_MASK <= 7.
                const uint16_t raw1 =
                    static_cast<uint16_t>(v_tile[tok8 * HEAD_DIM + sw0]) |
                    (static_cast<uint16_t>(v_tile[tok9 * HEAD_DIM + sw1]) << 8);

                float tmp0 = 0.f, tmp1 = 0.f, tmp2 = 0.f, tmp3 = 0.f;
                mma_m16n8k16(tmp0, tmp1, tmp2, tmp3,
                             a0, a1, a2, a3,
                             fp8x2_to_u32_f16x2(raw0), fp8x2_to_u32_f16x2(raw1));
                o_acc[nt][0] += vs_val * tmp0;
                o_acc[nt][1] += vs_val * tmp1;
                o_acc[nt][2] += vs_val * tmp2;
                o_acc[nt][3] += vs_val * tmp3;
            }
        }

        __syncthreads();
        if (t + 2 < num_tiles) {
            issue_tile_cp_async(t + 2, stage);
        }
    }

    // ── Reduce rmax, rsum, and o_acc across the 4 warps into smem_kv (now free) ──
    if (tid_in_group == 0) {
        smem_red[warp_id * 16 + group_id]     = rsum_lo;
        smem_red[warp_id * 16 + group_id + 8] = rsum_hi;
        if (warp_id == 0) {
            smem_rmax[group_id]     = rmax_lo;
            smem_rmax[group_id + 8] = rmax_hi;
        }
    }

    float (*out_smem)[HEAD_DIM] = reinterpret_cast<float(*)[HEAD_DIM]>(smem_raw);
    __syncthreads();

    if (tid < m_valid) {
        smem_rsum[tid] = smem_red[0 * 16 + tid] + smem_red[1 * 16 + tid]
                       + smem_red[2 * 16 + tid] + smem_red[3 * 16 + tid];
    }

    for (int w = 0; w < TC_WARPS; w++) {
        if (warp_id == w) {
            #pragma unroll
            for (int nt = 0; nt < NUM_N8_TILES; nt++) {
                const int d0 = nt * 8 + 2 * tid_in_group;
                if (w == 0) {
                    out_smem[group_id][d0]         = o_acc[nt][0];
                    out_smem[group_id][d0 + 1]     = o_acc[nt][1];
                    out_smem[group_id + 8][d0]     = o_acc[nt][2];
                    out_smem[group_id + 8][d0 + 1] = o_acc[nt][3];
                } else {
                    out_smem[group_id][d0]         += o_acc[nt][0];
                    out_smem[group_id][d0 + 1]     += o_acc[nt][1];
                    out_smem[group_id + 8][d0]     += o_acc[nt][2];
                    out_smem[group_id + 8][d0 + 1] += o_acc[nt][3];
                }
            }
        }
        __syncthreads();
    }

    // Cooperative coalesced store to O_partial / m_partial / l_partial.
    for (int i = tid; i < m_valid * HEAD_DIM; i += TC_THREADS) {
        const int m       = i / HEAD_DIM;
        const int d       = i % HEAD_DIM;
        const int row_idx = m_base + m;
        const int s       = row_idx / G;
        const int g       = row_idx % G;
        const int64_t hs  = (static_cast<int64_t>(b * S_q + s) * Hq + (kv_head * G + g))
                          * num_splits + split;
        O_partial[hs * HEAD_DIM + d] = out_smem[m][d];
    }
    if (tid < m_valid) {
        const int row_idx = m_base + tid;
        const int s       = row_idx / G;
        const int g       = row_idx % G;
        const int64_t hs  = (static_cast<int64_t>(b * S_q + s) * Hq + (kv_head * G + g))
                          * num_splits + split;
        m_partial[hs] = smem_rmax[tid];
        l_partial[hs] = smem_rsum[tid];
    }
}

// ─── Pass 2: combine splits (supports B * S_q) ───────────────────────────────

__global__ void gqa_combine_tc_kernel(
    const float* __restrict__ O_partial,  // [B * S_q, Hq, num_splits, D]
    const float* __restrict__ m_partial,  // [B * S_q, Hq, num_splits]
    const float* __restrict__ l_partial,  // [B * S_q, Hq, num_splits]
    __half*      __restrict__ Out,        // [B * S_q, Hq, D]
    int Hq, int D, int num_splits)
{
    const int q_head = blockIdx.x;
    const int bs     = blockIdx.y;        // b * S_q + s
    const int tid    = threadIdx.x;       // channel [0..D-1]

    const int64_t hb = static_cast<int64_t>(bs) * Hq + q_head;
    const float* mp  = m_partial + hb * num_splits;
    const float* lp  = l_partial + hb * num_splits;
    const float* Op  = O_partial + hb * num_splits * D;

    float gmax = -FLT_MAX;
    for (int s = 0; s < num_splits; s++) {
        gmax = fmaxf(gmax, mp[s]);
    }

    float acc = 0.f, denom = 0.f;
    for (int s = 0; s < num_splits; s++) {
        const float m = mp[s];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - gmax);
        acc   += w * Op[static_cast<int64_t>(s) * D + tid];
        denom += w * lp[s];
    }

    const float inv = (denom > 1e-9f) ? (1.f / denom) : 0.f;
    Out[hb * D + tid] = __float2half_rn(acc * inv);
}

template <int HEAD_DIM, bool FP8>
int tc_smem_bytes() {
    constexpr int elem_bytes   = FP8 ? 1 : 2;
    constexpr int kv_bytes     = 2 * 2 * TC_BLOCK_N * HEAD_DIM * elem_bytes;
    constexpr int q_bytes      = 16 * HEAD_DIM * 2;
    constexpr int red_bytes    = (2 * TC_WARPS * 16 + 32) * static_cast<int>(sizeof(float));
    return kv_bytes + q_bytes + red_bytes;
}

template <int HEAD_DIM, bool FP8>
void launch_tc_impl(
    const __half* Q,
    const void* K, const void* V,
    const float* k_scale, const float* v_scale,
    float* O_partial, float* m_partial, float* l_partial,
    __half* Out,
    int B, int S_q, int N, int Hq, int Hkv, int G,
    int num_m_tiles, int num_splits, int split_len, int num_pages,
    float scale, bool causal)
{
    const int smem = tc_smem_bytes<HEAD_DIM, FP8>();
    auto kernel = gqa_decode_tc_kernel<HEAD_DIM, TC_BLOCK_N, FP8>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    dim3 gridA(Hkv * num_m_tiles, num_splits, B);
    kernel<<<gridA, dim3(TC_THREADS), smem>>>(
        Q, K, V, k_scale, v_scale, O_partial, m_partial, l_partial,
        B, S_q, N, Hq, Hkv, G, num_m_tiles, num_splits, split_len, num_pages,
        scale, causal ? 1 : 0);

    dim3 gridB(Hq, B * S_q);
    gqa_combine_tc_kernel<<<gridB, dim3(HEAD_DIM)>>>(
        O_partial, m_partial, l_partial, Out, Hq, HEAD_DIM, num_splits);
}

int choose_tc_splits(int sm_count, int blocks_per_sm, int Hkv, int num_m_tiles, int B, int N,
                     bool fp8, int& split_len_out)
{
    const int SPLIT_TOKENS = 512;
    const int MAX_SPLITS   = 512;
    const int TARGET_WAVES = 4;
    const int blocks_per_split = std::max(1, Hkv * num_m_tiles * B);
    const int capacity  = std::max(1, sm_count * blocks_per_sm);
    const int per_wave  = std::max(1, capacity / blocks_per_split);
    const int want      = (N + SPLIT_TOKENS - 1) / SPLIT_TOKENS;
    const int max_occ_splits = std::max(1, (TARGET_WAVES * capacity + blocks_per_split - 1) / blocks_per_split);
    const int capped_want = std::min(want, max_occ_splits);
    const int waves     = std::max(1, (capped_want + per_wave / 2) / per_wave);
    const int split_cap = std::max(1, N / TC_BLOCK_N);

    int num_splits = std::min(waves * per_wave, split_cap);
    num_splits = std::max(1, std::min(num_splits, MAX_SPLITS));
    int split_len = (N + num_splits - 1) / num_splits;
    if (fp8) {
        split_len = ((split_len + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS) * KV_PAGE_TOKENS;
    }
    num_splits = (N + split_len - 1) / split_len;
    split_len_out = split_len;
    return num_splits;
}

template <int HEAD_DIM, bool FP8>
int query_blocks_per_sm() {
    const int smem = tc_smem_bytes<HEAD_DIM, FP8>();
    auto kernel = gqa_decode_tc_kernel<HEAD_DIM, TC_BLOCK_N, FP8>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, TC_THREADS, smem);
    return bps > 0 ? bps : 1;
}

}  // namespace

// ─── public host launchers ───────────────────────────────────────────────────

torch::Tensor launch_fused_gqa_tc(
    torch::Tensor Q,     // [B, Hq, D] or [B, S_q, Hq, D] fp16
    torch::Tensor K,     // [B, N, Hkv, D]                  fp16
    torch::Tensor V,     // [B, N, Hkv, D]                  fp16
    double scale,
    bool causal)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat16 &&
                K.dtype() == torch::kFloat16 &&
                V.dtype() == torch::kFloat16, "all inputs must be fp16");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(Q.dim() == 3 || Q.dim() == 4, "Q must be [B, Hq, D] or [B, S_q, Hq, D]");
    TORCH_CHECK(K.dim() == 4 && V.dim() == 4, "K/V must be [B, N, Hkv, D]");

    const bool is_4d = (Q.dim() == 4);
    const int B   = Q.size(0);
    const int S_q = is_4d ? Q.size(1) : 1;
    const int Hq  = is_4d ? Q.size(2) : Q.size(1);
    const int D   = is_4d ? Q.size(3) : Q.size(2);
    const int N   = K.size(1);
    const int Hkv = K.size(2);

    TORCH_CHECK(S_q >= 1, "S_q must be >= 1");
    TORCH_CHECK(D == 64 || D == 128, "tc kernel requires head_dim in {64, 128}, got ", D);
    TORCH_CHECK(K.size(0) == B && V.size(0) == B, "batch size mismatch across Q/K/V");
    TORCH_CHECK(K.size(3) == D && V.size(3) == D, "K/V head_dim must equal ", D);
    TORCH_CHECK(V.size(1) == N && V.size(2) == Hkv, "V shape must match K");
    TORCH_CHECK(Hq % Hkv == 0, "Hq=", Hq, " must be divisible by Hkv=", Hkv);
    if (causal) {
        TORCH_CHECK(N >= S_q, "causal mode requires N (", N, ") >= S_q (", S_q, ")");
    }

    auto Out = is_4d ? torch::empty({B, S_q, Hq, D}, Q.options())
                     : torch::empty({B, Hq, D}, Q.options());
    if (N == 0) {
        Out.zero_();
        return Out;
    }

    const int G = Hq / Hkv;
    const int num_m_tiles = (G * S_q + 15) / 16;

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, Q.device().index());
    if (sm_count <= 0) sm_count = 132;

    const int bps = (D == 64) ? query_blocks_per_sm<64, false>()
                              : query_blocks_per_sm<128, false>();
    int split_len = 0;
    const int num_splits = choose_tc_splits(sm_count, bps, Hkv, num_m_tiles, B, N, false, split_len);

    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto O_partial = torch::empty({B * S_q, Hq, num_splits, D}, f32);
    auto m_partial = torch::empty({B * S_q, Hq, num_splits}, f32);
    auto l_partial = torch::empty({B * S_q, Hq, num_splits}, f32);

    const __half* q_ptr  = reinterpret_cast<const __half*>(Q.data_ptr<at::Half>());
    const void*   k_ptr  = K.data_ptr<at::Half>();
    const void*   v_ptr  = V.data_ptr<at::Half>();
    __half*       o_ptr  = reinterpret_cast<__half*>(Out.data_ptr<at::Half>());

    if (D == 64) {
        launch_tc_impl<64, false>(
            q_ptr, k_ptr, v_ptr, nullptr, nullptr,
            O_partial.data_ptr<float>(), m_partial.data_ptr<float>(), l_partial.data_ptr<float>(),
            o_ptr, B, S_q, N, Hq, Hkv, G, num_m_tiles, num_splits, split_len, 0,
            static_cast<float>(scale), causal);
    } else {
        launch_tc_impl<128, false>(
            q_ptr, k_ptr, v_ptr, nullptr, nullptr,
            O_partial.data_ptr<float>(), m_partial.data_ptr<float>(), l_partial.data_ptr<float>(),
            o_ptr, B, S_q, N, Hq, Hkv, G, num_m_tiles, num_splits, split_len, 0,
            static_cast<float>(scale), causal);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gqa_decode_tc launch failed: ", cudaGetErrorString(err));
    return Out;
}

torch::Tensor launch_fused_gqa_tc_fp8(
    torch::Tensor Q,        // [B, Hq, D] or [B, S_q, Hq, D] fp16
    torch::Tensor Kq,       // [B, N, Hkv, D]                  uint8 (e4m3)
    torch::Tensor Vq,       // [B, N, Hkv, D]                  uint8 (e4m3)
    torch::Tensor k_scale,  // [B, num_pages, Hkv]             fp32
    torch::Tensor v_scale,  // [B, num_pages, Hkv]             fp32
    double scale,
    bool causal)
{
    TORCH_CHECK(Q.is_cuda() && Kq.is_cuda() && Vq.is_cuda(), "Q/Kq/Vq must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be fp16");
    TORCH_CHECK(Kq.dtype() == torch::kUInt8 && Vq.dtype() == torch::kUInt8,
                "Kq/Vq must be uint8 (e4m3 bits)");
    TORCH_CHECK(k_scale.dtype() == torch::kFloat32 && v_scale.dtype() == torch::kFloat32,
                "k_scale/v_scale must be fp32");
    TORCH_CHECK(Q.is_contiguous() && Kq.is_contiguous() && Vq.is_contiguous() &&
                k_scale.is_contiguous() && v_scale.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(Q.dim() == 3 || Q.dim() == 4, "Q must be [B, Hq, D] or [B, S_q, Hq, D]");
    TORCH_CHECK(Kq.dim() == 4 && Vq.dim() == 4, "Kq/Vq must be [B, N, Hkv, D]");

    const bool is_4d = (Q.dim() == 4);
    const int B   = Q.size(0);
    const int S_q = is_4d ? Q.size(1) : 1;
    const int Hq  = is_4d ? Q.size(2) : Q.size(1);
    const int D   = is_4d ? Q.size(3) : Q.size(2);
    const int N   = Kq.size(1);
    const int Hkv = Kq.size(2);

    TORCH_CHECK(S_q >= 1, "S_q must be >= 1");
    TORCH_CHECK(D == 64 || D == 128, "tc-fp8 kernel requires head_dim in {64, 128}, got ", D);
    TORCH_CHECK(Kq.size(0) == B && Vq.size(0) == B, "batch size mismatch");
    TORCH_CHECK(Kq.size(3) == D && Vq.size(3) == D, "Kq/Vq head_dim must equal ", D);
    TORCH_CHECK(Vq.size(1) == N && Vq.size(2) == Hkv, "Vq shape must match Kq");
    TORCH_CHECK(Hq % Hkv == 0, "Hq=", Hq, " must be divisible by Hkv=", Hkv);
    if (causal) {
        TORCH_CHECK(N >= S_q, "causal mode requires N (", N, ") >= S_q (", S_q, ")");
    }

    const int num_pages = (N + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS;
    TORCH_CHECK(k_scale.size(0) == B && v_scale.size(0) == B &&
                k_scale.size(1) == num_pages && v_scale.size(1) == num_pages &&
                k_scale.size(2) == Hkv && v_scale.size(2) == Hkv,
                "k_scale/v_scale must have shape [B, ceil(N/64), Hkv]");

    auto Out = is_4d ? torch::empty({B, S_q, Hq, D}, Q.options())
                     : torch::empty({B, Hq, D}, Q.options());
    if (N == 0) {
        Out.zero_();
        return Out;
    }

    const int G = Hq / Hkv;
    const int num_m_tiles = (G * S_q + 15) / 16;

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, Q.device().index());
    if (sm_count <= 0) sm_count = 132;

    const int bps = (D == 64) ? query_blocks_per_sm<64, true>()
                              : query_blocks_per_sm<128, true>();
    int split_len = 0;
    const int num_splits = choose_tc_splits(sm_count, bps, Hkv, num_m_tiles, B, N, true, split_len);

    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto O_partial = torch::empty({B * S_q, Hq, num_splits, D}, f32);
    auto m_partial = torch::empty({B * S_q, Hq, num_splits}, f32);
    auto l_partial = torch::empty({B * S_q, Hq, num_splits}, f32);

    const __half* q_ptr  = reinterpret_cast<const __half*>(Q.data_ptr<at::Half>());
    const void*   k_ptr  = Kq.data_ptr<uint8_t>();
    const void*   v_ptr  = Vq.data_ptr<uint8_t>();
    __half*       o_ptr  = reinterpret_cast<__half*>(Out.data_ptr<at::Half>());

    if (D == 64) {
        launch_tc_impl<64, true>(
            q_ptr, k_ptr, v_ptr, k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
            O_partial.data_ptr<float>(), m_partial.data_ptr<float>(), l_partial.data_ptr<float>(),
            o_ptr, B, S_q, N, Hq, Hkv, G, num_m_tiles, num_splits, split_len, num_pages,
            static_cast<float>(scale), causal);
    } else {
        launch_tc_impl<128, true>(
            q_ptr, k_ptr, v_ptr, k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
            O_partial.data_ptr<float>(), m_partial.data_ptr<float>(), l_partial.data_ptr<float>(),
            o_ptr, B, S_q, N, Hq, Hkv, G, num_m_tiles, num_splits, split_len, num_pages,
            static_cast<float>(scale), causal);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gqa_decode_tc_fp8 launch failed: ", cudaGetErrorString(err));
    return Out;
}

std::vector<int64_t> gqa_tc_kernel_info(int64_t head_dim, bool fp8) {
    TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128");
    cudaFuncAttributes attr{};
    int smem = 0;
    int bps = 0;
    if (head_dim == 64 && !fp8) {
        auto k = gqa_decode_tc_kernel<64, TC_BLOCK_N, false>;
        cudaFuncGetAttributes(&attr, k);
        smem = tc_smem_bytes<64, false>();
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k, TC_THREADS, smem);
    } else if (head_dim == 64 && fp8) {
        auto k = gqa_decode_tc_kernel<64, TC_BLOCK_N, true>;
        cudaFuncGetAttributes(&attr, k);
        smem = tc_smem_bytes<64, true>();
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k, TC_THREADS, smem);
    } else if (head_dim == 128 && !fp8) {
        auto k = gqa_decode_tc_kernel<128, TC_BLOCK_N, false>;
        cudaFuncGetAttributes(&attr, k);
        smem = tc_smem_bytes<128, false>();
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k, TC_THREADS, smem);
    } else {
        auto k = gqa_decode_tc_kernel<128, TC_BLOCK_N, true>;
        cudaFuncGetAttributes(&attr, k);
        smem = tc_smem_bytes<128, true>();
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k, TC_THREADS, smem);
    }
    return {static_cast<int64_t>(attr.numRegs),
            static_cast<int64_t>(attr.localSizeBytes),
            static_cast<int64_t>(smem),
            static_cast<int64_t>(bps),
            static_cast<int64_t>(TC_THREADS)};
}
