#pragma once
// Shared contract between the FP8 KV quantiser (kv_fp8.cu) and the FP8 decode
// kernel (gqa_decode.cu). These must agree or the scales are read for the wrong
// tokens, so they live in one place.

// Tokens per quantisation page. One scale per (page, kv_head) per tensor.
// MUST equal TILE_V4 in gqa_decode.cu: the decode kernel assumes a tile is
// exactly one page, so it can load a single scale per tile instead of a
// per-token lookup. The launcher rounds split_len up to a multiple of this so
// tiles stay page-aligned.
#define KV_PAGE_TOKENS 64

// Largest finite magnitude representable in e4m3 (S.EEEE.MMM, bias 7).
#define E4M3_MAX 448.0f
