# Triton GQA decode

**Can a portable Triton kernel match or beat a hand-tuned CUDA decode kernel, in fp16 and with an FP8 KV cache?**

## The problem

Hand-written CUDA decode kernels lock an inference stack to one vendor and one
architecture. This repo's own kernel (`csrc/kernels/gqa_decode.cu`,
`gqa_decode_v4_kernel` and `gqa_decode_v4_fp8_kernel`) plateaued at 56% of H100
HBM peak at 64K context ([`RESULTS.md`](../RESULTS.md)). Its comments document why:
after removing 4x redundant KV traffic and a bank conflict, it is instruction-issue
bound, because QK and PV are scalar FMA chains, and "tensor-core MMA is the lever".
`gqa_decode.py` pulls that lever in Triton, which runs on any GPU Triton supports.

## Design

Layouts match the CUDA kernel: `q [B, Hq, D]` fp16, `k, v [B, N, Hkv, D]`,
output `[B, Hq, D]` fp16, `D = 128`.

- **Heads packed into `tl.dot`'s M dimension.** The G = Hq / Hkv query heads that
  share a KV head form one `[BLOCK_H, D]` tile, so each K/V tile is read from HBM
  once and multiplied for all G heads in a single `tl.dot` (QK and PV both run on
  tensor cores, with fp32 accumulation). `tl.dot` needs M >= 16, so
  `BLOCK_H = max(16, next_pow2(G))`; for G = 4 that is 12 padding rows, which costs
  nothing in a memory-bound kernel.
- **Split-KV.** At batch 1 there are only `B * Hkv` = 8 (batch, KV head) pairs, far
  too few programs to fill a GPU. The KV range is split across programs, each
  emitting an unnormalised partial output plus its running max and sum; a second
  kernel merges the partials with an exact log-sum-exp. The split count targets
  about two programs per SM and splits are page-aligned (see FP8).
- **Online softmax in base 2.** `log2(e)` is folded into the score scale so the
  loop uses `exp2`; partial maxima stay in that domain and only the combine kernel
  reads them.
- **FP8 (e4m3).** K and V are uint8 e4m3 bits with one fp32 scale per (64-token
  page, KV head) for each of K and V, shape `[B, pages, Hkv]` (convention of
  `csrc/kernels/kv_fp8.cu`; `quantize_kv_fp8` is a torch mirror of it). A tile
  never straddles a page (`BLOCK_N` divides 64, splits are multiples of 64), so a
  tile has exactly one scale. The e4m3 tile is converted to fp16 in registers
  (exact), and because the scale is constant over the tile it factors out of the
  dots: scores are scaled by `softmax_scale * k_scale`, and each tile's PV product
  by `v_scale` before it joins the accumulator. That equals dequantising first, at
  the cost of two small multiplies instead of one per K/V element.
- **Autotune** over `BLOCK_N` in {32, 64, 128}, `num_warps` in {4, 8},
  `num_stages` in {2, 3}, keyed on the batch-times-context size and dtype path.

Limits: head_dim 128, one query token, contiguous KV (no paging, masks or
variable lengths). FP8 needs a GPU with e4m3 support in Triton (sm_89 or newer).

## Run

```bash
# correctness, on a GPU or on CPU under the Triton interpreter (no FP8 hardware needed)
python -m pytest tests/test_triton_decode.py -q
TRITON_INTERPRET=1 python -m pytest tests/test_triton_decode.py -q

# optional: CUDA extension for the v4 comparison (CUDA_ARCH=sm_89 for an RTX 40-series, default sm_90a)
CUDA_ARCH=sm_89 python setup.py build_ext --inplace

# benchmark -> results/triton/<gpu>/decode.{csv,md}; --peak-gbs is required except on H100 SXM
python benchmarks/bench_triton_decode.py --peak-gbs <GB/s>

# Nsight Compute: one 32K-context call of CUDA v4 and of the Triton kernel
python benchmarks/profile_ncu.py --ncu --ctx 32768

# H100 on Modal
modal run modal_apps/bench_triton_h100.py
```

The benchmark times each call with a cold L2 and reports effective bandwidth as
(Q + K + V + scales + O bytes) / time. The Nsight run collects DRAM throughput,
issue-slot utilisation, tensor-pipe (HMMA) activity, achieved warps and duration,
to test whether the CUDA kernel is issue-bound and whether Triton moves the work
onto tensor cores.

## Results

Pending measurement. Nothing has been run on a GPU yet; the tables from
`results/triton/` will be added here once they have.
