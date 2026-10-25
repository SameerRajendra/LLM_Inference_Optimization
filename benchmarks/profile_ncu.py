"""
Single-launch driver for Nsight Compute.

profile_kernel_vs_sdpa.py runs ~226 launches per context, which makes `ncu`
crawl. This runs ONE shape, warms up, then brackets exactly one call with
cudaProfilerStart/Stop. Bracketing (rather than `ncu -s N`) matters for Triton:
its first call autotunes by launching the kernel dozens of times, which would
shift any launch-skip count onto an arbitrary config.

One command profiles the CUDA v4 kernel and the Triton kernel at one context and
prints the counters that test "v4 is instruction-issue bound; Triton moves the
work onto tensor cores":

    python benchmarks/profile_ncu.py --ncu --ctx 32768

Read it as: sm__inst_issued high and tensor-pipe (hmma) ~0 for v4; tensor-pipe
active and lower issue pressure for Triton. ncu locks clocks and flushes caches,
so its durations are for comparing the two kernels with each other, not for
reporting (use bench_triton_decode.py for timings).

A single implementation, for a custom ncu command line (add --impl sdpa to
profile PyTorch SDPA):

    ncu --profile-from-start off -c 1 --section SpeedOfLight \
        python benchmarks/profile_ncu.py --impl v3 --ctx 64000

On WSL2 the counters are blocked until Windows allows them: NVIDIA Control Panel
> Desktop > Enable Developer Settings, then Developer > Manage GPU Performance
Counters > Allow access to all users.

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HQ, HKV, D = 32, 8, 128

# ncu kernel-name regex for each implementation.
KERNELS = {"v3": "gqa_decode_splitkv_kernel", "v4": "gqa_decode_v4_kernel",
           "triton": "_decode_split_kernel"}
METRICS = [
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_issued.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


def make_call(impl, q, k, v):
    scale = D ** -0.5
    if impl == "triton":
        from triton_kernels.gqa_decode import gqa_decode
        return lambda: gqa_decode(q, k, v)
    if impl == "sdpa":
        qs, ks, vs = q.unsqueeze(2), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
        return lambda: F.scaled_dot_product_attention(qs, ks, vs, scale=scale, enable_gqa=True)
    import sparse_kv._C as _C
    fn = {"v3": _C.fused_gqa, "v4": _C.fused_gqa_v4}[impl]
    return lambda: fn(q, k, v, scale)


def run_ncu(args):
    for impl in ("v4", "triton"):
        print("== {} ({}), ctx={}".format(impl, KERNELS[impl], args.ctx), flush=True)
        subprocess.run(
            [args.ncu_bin, "--profile-from-start", "off", "-k", "regex:" + KERNELS[impl],
             "-c", "1", "--metrics", ",".join(METRICS),
             sys.executable, os.path.abspath(__file__), "--impl", impl,
             "--ctx", str(args.ctx), "--batch", str(args.batch), "--warmup", str(args.warmup)],
            check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3, help="calls before the profiled one")
    ap.add_argument("--impl", choices=["v3", "v4", "triton", "sdpa"], default="v4")
    ap.add_argument("--ncu", action="store_true",
                    help="run ncu on the v4 and Triton kernels and print their counters")
    ap.add_argument("--ncu-bin", default="ncu")
    args = ap.parse_args()
    if args.ncu:
        return run_ncu(args)

    torch.manual_seed(0)
    q = torch.randn(args.batch, HQ, D, device="cuda", dtype=torch.float16)
    k = torch.randn(args.batch, args.ctx, HKV, D, device="cuda", dtype=torch.float16)
    v = torch.randn(args.batch, args.ctx, HKV, D, device="cuda", dtype=torch.float16)
    fn = make_call(args.impl, q, k, v)

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print("profiled {} | batch={} ctx={} | unique KV {:.3f} GB".format(
        args.impl, args.batch, args.ctx, 2 * k.numel() * 2 / 1e9))


if __name__ == "__main__":
    main()
