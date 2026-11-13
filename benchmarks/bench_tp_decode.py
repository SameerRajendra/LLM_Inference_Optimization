"""Multi-GPU TP=2/4 decode layer overlap & CUDA Graph benchmark (WP-5 Task 3).

Measures a single transformer decode layer partitioned across TP ranks using
torch.distributed (NCCL AllReduce) and sparse_kv.ops paged KV attention:
  - Column-parallel QKV projection (Hq/TP, Hkv/TP)
  - Paged KV cache append + decode (paged_fp16 or paged_fp8 via sparse_kv.ops)
  - Row-parallel O_Proj + NCCL AllReduce (TP_AllReduce_Attn)
  - SwiGLU MLP (column-parallel gate+up, row-parallel down)
    + NCCL AllReduce (TP_AllReduce_MLP)

Compares across (batch x context):
  1. Compute-only latency (same ops with dist.all_reduce skipped)
  2. Eager TP latency (compute + NCCL AllReduce)
  3. CUDA Graph TP latency (torch.cuda.CUDAGraph captured over NCCL AllReduce,
     guarded by NCCL >= 2.19 and clean fallback if unsupported)

Launch modes:
  Standalone (spawns TP workers via torch.multiprocessing.spawn + local TCP store):
    python benchmarks/bench_tp_decode.py --tp 2 --shape qwen3-8b

  torchrun:
    torchrun --nproc_per_node=4 benchmarks/bench_tp_decode.py --shape qwen3-8b

  Nsight Systems capture:
    nsys profile --capture-range=cudaProfilerApi -o profiles/tp_decode \
        torchrun --nproc_per_node=2 benchmarks/bench_tp_decode.py \
        --nsys-bracket --steps 16

  CPU dry-run / unit validation:
    python benchmarks/bench_tp_decode.py --dry-run-cpu --shape qwen3-8b --tp 2

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
import contextlib
import gc
import math
import os
import socket
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from bench_common import save_result  # noqa: E402

SHAPES = {
    "qwen3-8b": {
        "hidden": 4096,
        "intermediate": 12288,
        "hq": 32,
        "hkv": 8,
        "d": 128,
    },
    "llama3-8b": {
        "hidden": 4096,
        "intermediate": 14336,
        "hq": 32,
        "hkv": 8,
        "d": 128,
    },
    "tiny": {
        "hidden": 1024,
        "intermediate": 2816,
        "hq": 8,
        "hkv": 2,
        "d": 128,
    },
}

KV_MODES = ("paged_fp16", "paged_fp8")


def compute_tp_partition(shape_name: str, tp_size: int) -> Dict[str, Any]:
    """Compute and validate per-rank tensor-parallel dimensions for a model shape.

    Pure Python helper (does not require torch/CUDA) for CLI checks and unit tests.
    """
    if shape_name not in SHAPES:
        raise ValueError(
            "Unknown shape {!r}; choose from {}".format(
                shape_name, tuple(SHAPES.keys())
            )
        )
    if not isinstance(tp_size, int) or isinstance(tp_size, bool) or tp_size < 1:
        raise ValueError("tp_size must be a positive integer, got {!r}".format(tp_size))

    cfg = SHAPES[shape_name]
    hq = cfg["hq"]
    hkv = cfg["hkv"]
    d = cfg["d"]
    hidden = cfg["hidden"]
    intermediate = cfg["intermediate"]

    if hkv % tp_size != 0:
        raise ValueError(
            "Shape {!r} has Hkv={}, which is not divisible by tp_size={}".format(
                shape_name, hkv, tp_size
            )
        )
    if hq % tp_size != 0:
        raise ValueError(
            "Shape {!r} has Hq={}, which is not divisible by tp_size={}".format(
                shape_name, hq, tp_size
            )
        )
    if intermediate % tp_size != 0:
        raise ValueError(
            "Shape {!r} has intermediate={}, not divisible by tp_size={}".format(
                shape_name, intermediate, tp_size
            )
        )

    hq_local = hq // tp_size
    hkv_local = hkv // tp_size
    intermediate_local = intermediate // tp_size

    return {
        "shape": shape_name,
        "tp_size": tp_size,
        "hidden": hidden,
        "intermediate": intermediate,
        "hq": hq,
        "hkv": hkv,
        "d": d,
        "hq_local": hq_local,
        "hkv_local": hkv_local,
        "q_dim_local": hq_local * d,
        "k_dim_local": hkv_local * d,
        "v_dim_local": hkv_local * d,
        "qkv_dim_local": (hq_local + 2 * hkv_local) * d,
        "o_in_dim": hq_local * d,
        "o_out_dim": hidden,
        "intermediate_local": intermediate_local,
        "gate_up_dim_local": 2 * intermediate_local,
        "down_in_dim": intermediate_local,
        "down_out_dim": hidden,
    }


def find_free_port() -> int:
    """Bind an ephemeral localhost TCP port for standalone torch.distributed init."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def nccl_version(torch_mod: Any) -> Tuple[int, int, int]:
    """Return NCCL version as (major, minor, patch), or (0, 0, 0) if unavailable."""
    try:
        ver = torch_mod.cuda.nccl.version()
        if isinstance(ver, (tuple, list)) and len(ver) >= 2:
            major = int(ver[0])
            minor = int(ver[1])
            patch = int(ver[2]) if len(ver) >= 3 else 0
            return (major, minor, patch)
        if isinstance(ver, int):
            if ver >= 10000:
                return (ver // 10000, (ver % 10000) // 100, ver % 100)
            return (ver // 1000, (ver % 1000) // 100, ver % 100)
    except Exception:
        pass
    return (0, 0, 0)


def compute_tp_metrics(
    compute_ms: float,
    eager_tp_ms: float,
    graph_tp_ms: Optional[float],
) -> Dict[str, Optional[float]]:
    """Compute comm_ms, comm_pct, graph_speedup, and compute_to_comm_ratio."""
    comm_ms = eager_tp_ms - compute_ms
    comm_pct = (100.0 * comm_ms / eager_tp_ms) if eager_tp_ms > 0.0 else 0.0
    graph_speedup = (
        (eager_tp_ms / graph_tp_ms)
        if (graph_tp_ms is not None and graph_tp_ms > 0.0)
        else None
    )
    compute_to_comm_ratio = (compute_ms / comm_ms) if comm_ms > 0.0 else None
    return {
        "comm_ms": comm_ms,
        "comm_pct": comm_pct,
        "graph_speedup": graph_speedup,
        "compute_to_comm_ratio": compute_to_comm_ratio,
    }


def _parse_int_list(raw: str, name: str) -> List[int]:
    vals = []
    for item in raw.split(","):
        s = item.strip()
        if not s:
            continue
        v = int(s)
        if v <= 0:
            raise ValueError("{} entries must be > 0, got {}".format(name, v))
        vals.append(v)
    if not vals:
        raise ValueError("{} must not be empty".format(name))
    return vals


def _resolve_kv_modes(kv_mode: str) -> List[str]:
    if kv_mode == "both":
        return list(KV_MODES)
    return [kv_mode]


def _is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _print_dry_run_summary(
    part: Dict[str, Any],
    kv_modes: List[str],
    batches: List[int],
    contexts: List[int],
    block_size: int,
) -> None:
    print(
        "TP Partition | shape={shape} tp={tp_size} hidden={hidden} "
        "intermediate={intermediate} (local={intermediate_local}) "
        "Hq={hq} (local={hq_local}) Hkv={hkv} (local={hkv_local}) D={d}".format(
            **part
        )
    )
    print(
        "  QKV_Proj out_dim={qkv_dim_local} (Q={q_dim_local}, K={k_dim_local}, "
        "V={v_dim_local}) | O_Proj=({o_out_dim}, {o_in_dim}) | "
        "GateUp=({gate_up_dim_local}, {hidden}) | "
        "Down=({down_out_dim}, {down_in_dim})".format(**part)
    )
    print(
        "  Sweep: kv_modes={} batches={} contexts={} block_size={}".format(
            kv_modes, batches, contexts, block_size
        )
    )


def run_dry_run_cpu(
    shape_name: str,
    tp_size: int,
    kv_modes: List[str],
    batches: List[int],
    contexts: List[int],
    block_size: int,
) -> None:
    """Validate TP partitioning and run CPU shard-equivalence check if torch exists."""
    part = compute_tp_partition(shape_name, tp_size)
    _print_dry_run_summary(part, kv_modes, batches, contexts, block_size)

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("dry-run-cpu: torch not installed; partition validation passed.")
        return

    torch.manual_seed(0)
    b = min(batches[0], 4)
    ctx_test = min(contexts[0], 64)
    hidden = part["hidden"]
    hq = part["hq"]
    hkv = part["hkv"]
    d = part["d"]
    inter = part["intermediate"]
    scale = d ** -0.5

    x = torch.randn(b, hidden, dtype=torch.float32)
    ln1_w = torch.ones(hidden, dtype=torch.float32)
    ln2_w = torch.ones(hidden, dtype=torch.float32)
    w_q = torch.randn(hq * d, hidden, dtype=torch.float32) * 0.02
    w_k = torch.randn(hkv * d, hidden, dtype=torch.float32) * 0.02
    w_v = torch.randn(hkv * d, hidden, dtype=torch.float32) * 0.02
    w_o = torch.randn(hidden, hq * d, dtype=torch.float32) * 0.02
    w_gate = torch.randn(inter, hidden, dtype=torch.float32) * 0.02
    w_up = torch.randn(inter, hidden, dtype=torch.float32) * 0.02
    w_down = torch.randn(hidden, inter, dtype=torch.float32) * 0.02

    k_cache_full = torch.randn(b, ctx_test, hkv, d, dtype=torch.float32) * 0.02
    v_cache_full = torch.randn(b, ctx_test, hkv, d, dtype=torch.float32) * 0.02

    def rms_norm(t: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return w * t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)

    # Unsharded reference
    h0 = rms_norm(x, ln1_w)
    q_ref = F.linear(h0, w_q).view(b, hq, 1, d)
    k_new_ref = F.linear(h0, w_k).view(b, 1, hkv, d)
    v_new_ref = F.linear(h0, w_v).view(b, 1, hkv, d)
    k_all_ref = torch.cat([k_cache_full[:, :-1], k_new_ref], dim=1).transpose(1, 2)
    v_all_ref = torch.cat([v_cache_full[:, :-1], v_new_ref], dim=1).transpose(1, 2)
    attn_ref = F.scaled_dot_product_attention(
        q_ref, k_all_ref, v_all_ref, scale=scale, enable_gqa=True
    ).reshape(b, hq * d)
    h_mid_ref = x + F.linear(attn_ref, w_o)
    h2_ref = rms_norm(h_mid_ref, ln2_w)
    out_ref = h_mid_ref + F.linear(
        F.silu(F.linear(h2_ref, w_gate)) * F.linear(h2_ref, w_up), w_down
    )

    # Sharded TP simulation across tp_size ranks
    q_dim = part["q_dim_local"]
    kv_dim = part["k_dim_local"]
    inter_local = part["intermediate_local"]
    hkv_local = part["hkv_local"]
    hq_local = part["hq_local"]

    attn_partials = []
    for r in range(tp_size):
        wq_r = w_q[r * q_dim:(r + 1) * q_dim]
        wk_r = w_k[r * kv_dim:(r + 1) * kv_dim]
        wv_r = w_v[r * kv_dim:(r + 1) * kv_dim]
        wo_r = w_o[:, r * q_dim:(r + 1) * q_dim]

        q_r = F.linear(h0, wq_r).view(b, hq_local, 1, d)
        k_new_r = F.linear(h0, wk_r).view(b, 1, hkv_local, d)
        v_new_r = F.linear(h0, wv_r).view(b, 1, hkv_local, d)
        k_all_r = torch.cat(
            [k_cache_full[:, :-1, r * hkv_local:(r + 1) * hkv_local], k_new_r], dim=1
        ).transpose(1, 2)
        v_all_r = torch.cat(
            [v_cache_full[:, :-1, r * hkv_local:(r + 1) * hkv_local], v_new_r], dim=1
        ).transpose(1, 2)
        attn_r = F.scaled_dot_product_attention(
            q_r, k_all_r, v_all_r, scale=scale, enable_gqa=True
        ).reshape(b, q_dim)
        attn_partials.append(F.linear(attn_r, wo_r))

    h_mid_tp = x + sum(attn_partials)
    h2_tp = rms_norm(h_mid_tp, ln2_w)

    mlp_partials = []
    for r in range(tp_size):
        wg_r = w_gate[r * inter_local:(r + 1) * inter_local]
        wu_r = w_up[r * inter_local:(r + 1) * inter_local]
        wd_r = w_down[:, r * inter_local:(r + 1) * inter_local]
        mlp_partials.append(
            F.linear(F.silu(F.linear(h2_tp, wg_r)) * F.linear(h2_tp, wu_r), wd_r)
        )

    out_tp = h_mid_tp + sum(mlp_partials)
    max_err = float((out_tp - out_ref).abs().max().item())
    print(
        "dry-run-cpu: TP={} equivalence check passed "
        "(B={}, ctx={}, max_abs_err={:.3e})".format(tp_size, b, ctx_test, max_err)
    )
    if max_err > 1e-4:
        raise RuntimeError("CPU TP shard mismatch: max_abs_err={:.3e}".format(max_err))


@contextlib.contextmanager
def _nvtx_range(torch_mod: Any, name: str, enabled: bool):
    if enabled:
        torch_mod.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch_mod.cuda.nvtx.range_pop()


def _cuda_profiler_start(torch_mod: Any) -> None:
    try:
        rt = torch_mod.cuda.cudart()
        if rt is not None:
            rt.cudaProfilerStart()
    except Exception:
        pass


def _cuda_profiler_stop(torch_mod: Any) -> None:
    try:
        rt = torch_mod.cuda.cudart()
        if rt is not None:
            rt.cudaProfilerStop()
    except Exception:
        pass


def _rms_norm(x: Any, w: Any, eps: float = 1e-6) -> Any:
    xf = x.float()
    xf = xf * xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return w * xf.to(x.dtype)


class TPDecodeLayerState:
    """Per-rank weights, paged KV cache, and static buffers for (kv_mode, B, ctx)."""

    def __init__(
        self,
        torch_mod: Any,
        dist_mod: Any,
        kv_ops: Any,
        allocate_paged_kv_cache: Callable[..., Any],
        part: Dict[str, Any],
        kv_mode: str,
        batch_size: int,
        ctx: int,
        block_size: int,
        rank: int,
        device: Any,
    ) -> None:
        self.torch = torch_mod
        self.dist = dist_mod
        self.kv_ops = kv_ops
        self.part = part
        self.kv_mode = kv_mode
        self.batch_size = batch_size
        self.ctx = ctx
        self.block_size = block_size
        self.device = device
        self.eps = 1e-6
        self.scale = part["d"] ** -0.5

        hidden = part["hidden"]
        self.q_dim = part["q_dim_local"]
        self.kv_dim = part["k_dim_local"]
        self.hq_local = part["hq_local"]
        self.hkv_local = part["hkv_local"]
        self.d = part["d"]
        inter_local = part["intermediate_local"]

        gen = torch_mod.Generator(device="cpu").manual_seed(1000 + rank)

        def _w(rows: int, cols: int) -> Any:
            t = torch_mod.randn(
                rows, cols, generator=gen, dtype=torch_mod.float16
            ) * 0.02
            return t.to(device)

        self.ln1_w = torch_mod.ones(hidden, dtype=torch_mod.float16, device=device)
        self.ln2_w = torch_mod.ones(hidden, dtype=torch_mod.float16, device=device)
        self.w_qkv = _w(part["qkv_dim_local"], hidden)
        self.w_o = _w(hidden, self.q_dim)
        self.w_gate_up = _w(2 * inter_local, hidden)
        self.w_down = _w(hidden, inter_local)

        # Static activation and AllReduce buffers (pointer-stable for CUDAGraph).
        self.x = (
            torch_mod.randn(
                batch_size, hidden, generator=gen, dtype=torch_mod.float16
            )
            * 0.05
        ).to(device)
        self.attn_ar_buf = torch_mod.empty(
            batch_size, hidden, dtype=torch_mod.float16, device=device
        )
        self.mlp_ar_buf = torch_mod.empty(
            batch_size, hidden, dtype=torch_mod.float16, device=device
        )
        self.x_out = torch_mod.empty(
            batch_size, hidden, dtype=torch_mod.float16, device=device
        )

        # Paged KV pool for this rank's Hkv/TP shard.
        bps = (ctx + block_size - 1) // block_size
        num_blocks = batch_size * bps
        fp8 = kv_mode == "paged_fp8"
        (
            self.k_cache,
            self.v_cache,
            self.k_scale,
            self.v_scale,
        ) = allocate_paged_kv_cache(
            num_blocks, block_size, self.hkv_local, self.d, fp8=fp8, device=device
        )
        if fp8:
            # 0..59 in uint8 e4m3 are valid finite non-NaN magnitudes.
            self.k_cache.random_(0, 60)
            self.v_cache.random_(0, 60)
            self.k_scale.fill_(0.05)
            self.v_scale.fill_(0.05)
        else:
            self.k_cache.normal_(0.0, 0.02)
            self.v_cache.normal_(0.0, 0.02)

        bt_gen = torch_mod.Generator(device="cpu").manual_seed(42 + rank)
        self.block_table = (
            torch_mod.randperm(num_blocks, generator=bt_gen)
            .view(batch_size, bps)
            .to(dtype=torch_mod.int32, device=device)
            .contiguous()
        )
        self.seq_lens = torch_mod.full(
            (batch_size,), ctx, dtype=torch_mod.int32, device=device
        )
        last_pos = ctx - 1
        last_blk = self.block_table[:, last_pos // block_size].long()
        self.slot_mapping = (
            (last_blk * block_size + (last_pos % block_size))
            .to(dtype=torch_mod.int32)
            .contiguous()
        )

    def kv_bytes_per_rank(self) -> int:
        total = self.k_cache.numel() * self.k_cache.element_size()
        total += self.v_cache.numel() * self.v_cache.element_size()
        if self.k_scale is not None:
            total += self.k_scale.numel() * self.k_scale.element_size()
        if self.v_scale is not None:
            total += self.v_scale.numel() * self.v_scale.element_size()
        return int(total)

    def step(self, use_allreduce: bool, nvtx_enabled: bool = True) -> Any:
        torch_mod = self.torch
        F = torch_mod.nn.functional
        B = self.batch_size

        h = _rms_norm(self.x, self.ln1_w, self.eps)
        qkv = F.linear(h, self.w_qkv)
        q, k, v = qkv.split((self.q_dim, self.kv_dim, self.kv_dim), dim=-1)
        q = q.view(B, self.hq_local, self.d).contiguous()
        k = k.view(B, self.hkv_local, self.d).contiguous()
        v = v.view(B, self.hkv_local, self.d).contiguous()

        with _nvtx_range(torch_mod, "TP_Paged_Attn", nvtx_enabled):
            self.kv_ops.reshape_and_cache(
                k,
                v,
                self.k_cache,
                self.v_cache,
                self.slot_mapping,
                self.k_scale,
                self.v_scale,
            )
            attn_out = self.kv_ops.paged_gqa_decode(
                q,
                self.k_cache,
                self.v_cache,
                self.block_table,
                self.seq_lens,
                max_seq_len=self.ctx,
                scale=self.scale,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
            )

        o_proj_out = F.linear(attn_out.reshape(B, self.q_dim), self.w_o)
        self.attn_ar_buf.copy_(o_proj_out)
        if use_allreduce:
            with _nvtx_range(torch_mod, "TP_AllReduce_Attn", nvtx_enabled):
                self.dist.all_reduce(self.attn_ar_buf, op=self.dist.ReduceOp.SUM)
        h_mid = self.x + self.attn_ar_buf

        with _nvtx_range(torch_mod, "TP_MLP_SwiGLU", nvtx_enabled):
            h_norm2 = _rms_norm(h_mid, self.ln2_w, self.eps)
            gate_up = F.linear(h_norm2, self.w_gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            mlp_out = F.linear(F.silu(gate) * up, self.w_down)
            self.mlp_ar_buf.copy_(mlp_out)

        if use_allreduce:
            with _nvtx_range(torch_mod, "TP_AllReduce_MLP", nvtx_enabled):
                self.dist.all_reduce(self.mlp_ar_buf, op=self.dist.ReduceOp.SUM)

        self.x_out.copy_(h_mid + self.mlp_ar_buf)
        return self.x_out


def _time_mode(
    torch_mod: Any,
    dist_mod: Any,
    fn: Callable[[], Any],
    warmup: int,
    steps: int,
    nsys_bracket: bool,
    nvtx_enabled: bool,
) -> Dict[str, float]:
    dist_mod.barrier()
    with torch_mod.no_grad():
        for _ in range(warmup):
            fn()
        torch_mod.cuda.synchronize()
        dist_mod.barrier()

        starts = [torch_mod.cuda.Event(enable_timing=True) for _ in range(steps)]
        ends = [torch_mod.cuda.Event(enable_timing=True) for _ in range(steps)]

        if nsys_bracket:
            _cuda_profiler_start(torch_mod)
        t0 = time.perf_counter()
        for i in range(steps):
            with _nvtx_range(torch_mod, "TP_Decode_Step", nvtx_enabled):
                starts[i].record()
                fn()
                ends[i].record()
        torch_mod.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        if nsys_bracket:
            _cuda_profiler_stop(torch_mod)

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    n = len(times)
    p99_idx = min(n - 1, int(math.ceil(0.99 * n)) - 1)
    return {
        "mean_ms": sum(times) / n,
        "p50_ms": times[n // 2],
        "p99_ms": times[p99_idx],
        "wall_ms": (wall_s * 1e3) / n,
    }


def _worker_main(
    rank: int,
    world_size: int,
    local_rank: int,
    init_method: Optional[str],
    args: argparse.Namespace,
    launch_mode: str,
) -> None:
    import torch
    import torch.distributed as dist

    from sparse_kv import ops as kv_ops
    from triton_kernels.paged_kv import allocate_paged_kv_cache

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if init_method is not None:
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    else:
        dist.init_process_group(backend="nccl")

    part = compute_tp_partition(args.shape, world_size)
    batches = _parse_int_list(args.batches, "--batches")
    contexts = _parse_int_list(args.contexts, "--contexts")
    kv_modes = _resolve_kv_modes(args.kv_mode)
    nccl_ver = nccl_version(torch)
    nccl_graph_supported = nccl_ver >= (2, 19, 0)
    nccl_ver_str = "{}.{}.{}".format(*nccl_ver)

    if rank == 0:
        print(
            "TP Decode Benchmark | shape={} tp={} launch={} gpu={} nccl={}".format(
                args.shape,
                world_size,
                launch_mode,
                torch.cuda.get_device_name(device),
                nccl_ver_str,
            )
        )
        if not nccl_graph_supported:
            print(
                "NOTICE: NCCL {} < 2.19.0; skipping CUDA Graph capture over NCCL "
                "(requires NCCL >= 2.19).".format(nccl_ver_str)
            )

    rows = []
    noticed_graph_failure = False

    try:
        for kv_mode in kv_modes:
            for ctx in contexts:
                for batch_size in batches:
                    gc.collect()
                    torch.cuda.empty_cache()

                    oom = False
                    state = None
                    try:
                        state = TPDecodeLayerState(
                            torch_mod=torch,
                            dist_mod=dist,
                            kv_ops=kv_ops,
                            allocate_paged_kv_cache=allocate_paged_kv_cache,
                            part=part,
                            kv_mode=kv_mode,
                            batch_size=batch_size,
                            ctx=ctx,
                            block_size=args.block_size,
                            rank=rank,
                            device=device,
                        )
                    except torch.cuda.OutOfMemoryError:
                        oom = True

                    # Ensure every rank agrees if any rank OOMed during allocation.
                    oom_flag = torch.tensor(
                        [1 if oom else 0], dtype=torch.int32, device=device
                    )
                    dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
                    if int(oom_flag.item()) > 0:
                        state = None
                        gc.collect()
                        torch.cuda.empty_cache()
                        if rank == 0:
                            print(
                                "{:<10} B={:<3} ctx={:<6} skipped (OOM)".format(
                                    kv_mode, batch_size, ctx
                                )
                            )
                            rows.append(
                                {
                                    "shape": args.shape,
                                    "tp_size": world_size,
                                    "kv_mode": kv_mode,
                                    "batch": batch_size,
                                    "ctx": ctx,
                                    "skipped": "oom",
                                }
                            )
                        continue

                    assert state is not None
                    bracket_this = args.nsys_bracket and (
                        kv_mode == kv_modes[0]
                        and ctx == contexts[0]
                        and batch_size == batches[0]
                    )
                    if bracket_this:
                        _cuda_profiler_start(torch)

                    # 1. Compute-only (no dist.all_reduce)
                    with _nvtx_range(torch, "TP_Mode_Compute", True):
                        comp_stats = _time_mode(
                            torch,
                            dist,
                            lambda st=state: st.step(
                                use_allreduce=False, nvtx_enabled=True
                            ),
                            warmup=args.warmup,
                            steps=args.steps,
                            nsys_bracket=False,
                            nvtx_enabled=True,
                        )

                    # 2. Eager TP (compute + NCCL AllReduce)
                    with _nvtx_range(torch, "TP_Mode_Eager", True):
                        eager_stats = _time_mode(
                            torch,
                            dist,
                            lambda st=state: st.step(
                                use_allreduce=True, nvtx_enabled=True
                            ),
                            warmup=args.warmup,
                            steps=args.steps,
                            nsys_bracket=False,
                            nvtx_enabled=True,
                        )

                    # 3. CUDA Graph TP (captured over compute + NCCL AllReduce)
                    graph_stats = None
                    graph_obj = None
                    graph_status = "ok"
                    if not nccl_graph_supported:
                        graph_status = "skipped_nccl_lt_2.19 ({})".format(nccl_ver_str)
                    else:
                        dist.barrier()
                        try:
                            # Warm up Triton autotune and NCCL on side stream first.
                            graph_obj, _ = kv_ops.capture_cuda_graph(
                                lambda st=state: st.step(
                                    use_allreduce=True, nvtx_enabled=False
                                ),
                                warmup_iters=3,
                            )
                            with _nvtx_range(torch, "TP_Mode_Graph", True):
                                graph_stats = _time_mode(
                                    torch,
                                    dist,
                                    graph_obj.replay,
                                    warmup=args.warmup,
                                    steps=args.steps,
                                    nsys_bracket=False,
                                    nvtx_enabled=True,
                                )
                        except RuntimeError as exc:
                            first_line = str(exc).splitlines()[0]
                            graph_status = "unsupported: {}".format(first_line)
                            if rank == 0 and not noticed_graph_failure:
                                print(
                                    "NOTICE: CUDA Graph capture over NCCL unsupported "
                                    "on this stack ({}); falling back to eager TP "
                                    "only.".format(graph_status)
                                )
                                noticed_graph_failure = True

                    if bracket_this:
                        _cuda_profiler_stop(torch)

                    compute_ms = comp_stats["mean_ms"]
                    eager_tp_ms = eager_stats["mean_ms"]
                    graph_tp_ms = (
                        graph_stats["mean_ms"] if graph_stats is not None else None
                    )
                    derived = compute_tp_metrics(compute_ms, eager_tp_ms, graph_tp_ms)

                    row = {
                        "shape": args.shape,
                        "tp_size": world_size,
                        "kv_mode": kv_mode,
                        "batch": batch_size,
                        "ctx": ctx,
                        "steps": args.steps,
                        "compute_ms": compute_ms,
                        "compute_p50_ms": comp_stats["p50_ms"],
                        "compute_p99_ms": comp_stats["p99_ms"],
                        "eager_tp_ms": eager_tp_ms,
                        "eager_tp_p50_ms": eager_stats["p50_ms"],
                        "eager_tp_p99_ms": eager_stats["p99_ms"],
                        "graph_tp_ms": graph_tp_ms,
                        "graph_tp_p50_ms": (
                            graph_stats["p50_ms"] if graph_stats else None
                        ),
                        "graph_tp_p99_ms": (
                            graph_stats["p99_ms"] if graph_stats else None
                        ),
                        "graph_status": graph_status,
                        "comm_ms": derived["comm_ms"],
                        "comm_pct": derived["comm_pct"],
                        "graph_speedup": derived["graph_speedup"],
                        "compute_to_comm_ratio": derived["compute_to_comm_ratio"],
                        "kv_gb_per_rank": state.kv_bytes_per_rank() / 1e9,
                    }

                    if rank == 0:
                        rows.append(row)
                        speedup = derived["graph_speedup"]
                        if graph_tp_ms is not None and speedup is not None:
                            graph_str = "{:.3f} ms ({:.2f}x)".format(
                                graph_tp_ms, speedup
                            )
                        else:
                            graph_str = "skipped"
                        ratio = derived["compute_to_comm_ratio"]
                        ratio_str = (
                            "{:.2f}x".format(ratio) if ratio is not None else "inf"
                        )
                        print(
                            "{:<10} B={:<3} ctx={:<6} | compute {:6.3f} ms | "
                            "eager {:6.3f} ms | graph {:<18} | "
                            "comm {:6.3f} ms ({:4.1f}%) | comp/comm {}".format(
                                kv_mode,
                                batch_size,
                                ctx,
                                compute_ms,
                                eager_tp_ms,
                                graph_str,
                                derived["comm_ms"],
                                derived["comm_pct"],
                                ratio_str,
                            )
                        )

                    graph_obj = None
                    state = None
                    gc.collect()
                    torch.cuda.empty_cache()

        if rank == 0 and not args.no_save:
            extra = {
                "shape": args.shape,
                "tp_size": world_size,
                "launch_mode": launch_mode,
                "kv_modes": kv_modes,
                "batches": batches,
                "contexts": contexts,
                "steps": args.steps,
                "warmup": args.warmup,
                "block_size": args.block_size,
                "nccl_version": nccl_ver_str,
                "nccl_graph_supported": nccl_graph_supported,
                "partition": part,
                "gpu_count": torch.cuda.device_count(),
                "nsys_bracket": args.nsys_bracket,
            }
            label = args.label or "{}-tp{}".format(args.shape, world_size)
            save_result("tp_decode", rows, label=label, extra=extra)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_entry(
    rank: int,
    world_size: int,
    init_method: str,
    args: argparse.Namespace,
) -> None:
    _worker_main(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        init_method=init_method,
        args=args,
        launch_mode="mp.spawn",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Multi-GPU Tensor Parallel (TP=2/4) decode layer overlap "
            "& CUDA Graph benchmark."
        )
    )
    ap.add_argument(
        "--shape",
        choices=list(SHAPES.keys()),
        default="qwen3-8b",
        help="Model layer shape (default: qwen3-8b).",
    )
    ap.add_argument(
        "--tp",
        type=int,
        choices=[2, 4],
        default=None,
        help=(
            "Tensor parallel size (2 or 4; default 2 standalone, "
            "or WORLD_SIZE under torchrun)."
        ),
    )
    ap.add_argument(
        "--kv-mode",
        choices=["paged_fp16", "paged_fp8", "both"],
        default="paged_fp16",
        help="Paged KV cache format via sparse_kv.ops (default: paged_fp16).",
    )
    ap.add_argument(
        "--batches",
        default="1,4,16,64",
        help="Comma-separated batch sizes (default: 1,4,16,64).",
    )
    ap.add_argument(
        "--contexts",
        default="4096,16384",
        help="Comma-separated context lengths (default: 4096,16384).",
    )
    ap.add_argument(
        "--steps", type=int, default=64, help="Timed decode steps per config."
    )
    ap.add_argument(
        "--warmup", type=int, default=8, help="Warmup steps before timing."
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Paged KV block size (default: 64).",
    )
    ap.add_argument(
        "--nsys-bracket",
        action="store_true",
        help=(
            "Bracket --steps with cudaProfilerStart/Stop and TP_* "
            "NVTX ranges for nsys."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate TP partitioning and sweep configuration without GPUs.",
    )
    ap.add_argument(
        "--dry-run-cpu",
        action="store_true",
        help="Validate TP partitioning and run a CPU shard check without GPUs.",
    )
    ap.add_argument(
        "--label", default="", help="Optional result label for save_result()."
    )
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write JSON output under results/tp_decode/.",
    )
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    torchrun_mode = _is_torchrun()
    if torchrun_mode:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if args.tp is not None and args.tp != world_size:
            raise SystemExit(
                "Error: --tp {} disagrees with torchrun WORLD_SIZE={}".format(
                    args.tp, world_size
                )
            )
        if world_size not in (2, 4):
            raise SystemExit(
                "Error: TP decode benchmark supports WORLD_SIZE in {{2, 4}}, got "
                "{}".format(world_size)
            )
        tp_size = world_size
    else:
        tp_size = args.tp if args.tp is not None else 2
        rank = 0
        local_rank = 0

    try:
        part = compute_tp_partition(args.shape, tp_size)
        batches = _parse_int_list(args.batches, "--batches")
        contexts = _parse_int_list(args.contexts, "--contexts")
        kv_modes = _resolve_kv_modes(args.kv_mode)
    except ValueError as exc:
        raise SystemExit("Error: {}".format(exc))

    if args.block_size < 16 or (args.block_size & (args.block_size - 1)) != 0:
        raise SystemExit("Error: --block-size must be a power of 2 >= 16")
    if "paged_fp8" in kv_modes and args.block_size % 64 != 0:
        raise SystemExit("Error: paged_fp8 requires --block-size divisible by 64")

    if args.dry_run_cpu:
        run_dry_run_cpu(
            args.shape, tp_size, kv_modes, batches, contexts, args.block_size
        )
        return
    if args.dry_run:
        _print_dry_run_summary(part, kv_modes, batches, contexts, args.block_size)
        return

    try:
        import torch
        import torch.multiprocessing as mp
    except ImportError as exc:
        raise SystemExit(
            "Error: PyTorch with CUDA support is required to run GPU TP decode ({}). "
            "Use --dry-run-cpu for CPU validation.".format(exc)
        )

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count < tp_size:
        msg = (
            "Error: TP={} requires at least {} CUDA GPUs, but found {} "
            "(torch.cuda.is_available()={}). Pass --dry-run or --dry-run-cpu "
            "to validate shapes without GPUs.".format(
                tp_size, tp_size, gpu_count, torch.cuda.is_available()
            )
        )
        if rank == 0:
            print(msg, file=sys.stderr)
        raise SystemExit(1)

    if torchrun_mode:
        _worker_main(
            rank=rank,
            world_size=tp_size,
            local_rank=local_rank,
            init_method=None,
            args=args,
            launch_mode="torchrun",
        )
    else:
        port = find_free_port()
        init_method = "tcp://127.0.0.1:{}".format(port)
        mp.spawn(
            _spawn_entry,
            args=(tp_size, init_method, args),
            nprocs=tp_size,
            join=True,
        )


if __name__ == "__main__":
    main()
