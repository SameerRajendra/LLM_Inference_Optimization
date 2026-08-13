"""
Minimal single-launch driver for Nsight Compute.

profile_kernel_vs_sdpa.py runs ~226 launches per context, which makes `ncu`
crawl and forces fragile --launch-skip arithmetic to reach the shape you care
about. This runs ONE shape, with a fixed number of warmups, so the profiled
launch is deterministic:

    ncu -k regex:gqa_decode_splitkv -s 3 -c 1 \
        --section SpeedOfLight --section Occupancy \
        --section WarpStateStats --section MemoryWorkloadAnalysis \
        "$PYTHON_BIN" benchmarks/profile_ncu.py --ctx 64000

  -s 3  skips the 3 warmup launches
  -c 1  profiles exactly the 4th (the one after warmup)

Add --sdpa to profile the PyTorch/FlashAttention baseline under the same
harness for a like-for-like counter comparison.

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparse_kv._C import fused_gqa   # noqa: E402

HQ, HKV, D = 32, 8, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=64000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3,
                    help="launches before the one you profile (match ncu -s)")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--sdpa", action="store_true",
                    help="profile PyTorch SDPA instead of our kernel")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    B, N = args.batch, args.ctx
    scale = 1.0 / (D ** 0.5)

    Q = torch.randn(B, HQ, D, device=dev, dtype=torch.float16)
    Kc = torch.randn(B, N, HKV, D, device=dev, dtype=torch.float16)
    Vc = torch.randn(B, N, HKV, D, device=dev, dtype=torch.float16)

    if args.sdpa:
        Qs = Q.unsqueeze(2)
        Ks = Kc.transpose(1, 2).contiguous()
        Vs = Vc.transpose(1, 2).contiguous()
        try:
            F.scaled_dot_product_attention(Qs, Ks, Vs, scale=scale, enable_gqa=True)

            def fn():
                return F.scaled_dot_product_attention(
                    Qs, Ks, Vs, scale=scale, enable_gqa=True)
        except TypeError:
            n = HQ // HKV
            Ke = Ks[:, :, None].expand(B, HKV, n, N, D).reshape(B, HQ, N, D).contiguous()
            Ve = Vs[:, :, None].expand(B, HKV, n, N, D).reshape(B, HQ, N, D).contiguous()

            def fn():
                return F.scaled_dot_product_attention(Qs, Ke, Ve, scale=scale)
    else:
        def fn():
            return fused_gqa(Q, Kc, Vc, scale)

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    for _ in range(args.iters):
        fn()
    torch.cuda.synchronize()

    kv_gb = 2 * B * N * HKV * D * 2 / 1e9
    print("profiled {} | batch={} ctx={} | unique KV {:.3f} GB | "
          "warmup={} iters={}".format(
              "SDPA" if args.sdpa else "fused_gqa",
              B, N, kv_gb, args.warmup, args.iters))


if __name__ == "__main__":
    main()
