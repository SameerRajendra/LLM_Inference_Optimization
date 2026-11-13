"""PyTorch custom ops for the decode kernels, and CUDA-graph capture helpers (WP-2).

Registering the kernels with torch.library makes them opaque, schema-checked nodes:
torch.compile(fullgraph=True) traces through them without graph breaks (the fake impls give it
output shapes), and torch.cuda.CUDAGraph can capture them like any aten op.

    torch.ops.sparse_kv.fused_gqa_v4(q, k, v, scale)                       CUDA extension
    torch.ops.sparse_kv.fused_gqa_v4_fp8(q, kq, vq, k_scale, v_scale, scale)
    torch.ops.sparse_kv.triton_gqa_decode(q, k, v, scale, k_scale, v_scale, num_splits)
    torch.ops.sparse_kv.paged_gqa_decode(q, k_cache, v_cache, block_table, seq_lens, scale,
                                         k_scale, v_scale, max_seq_len, num_splits)
    torch.ops.sparse_kv.reshape_and_cache(k_new, v_new, k_cache, v_cache, slot_mapping)
    torch.ops.sparse_kv.reshape_and_cache_fp8(k_new, v_new, k_cache, v_cache, k_scale, v_scale,
                                              slot_mapping)

num_splits == 0 means "use the kernel's heuristic". paged_gqa_decode takes max_seq_len as a
required int: it sizes the split grid on the host, so it can never fall back to the
seq_lens.max().item() device -> host sync (which would also be illegal inside graph capture).

CUDA graphs: Triton's @autotune benchmarks configs (and synchronizes) the first time it sees a
key; that must not happen during capture. paged_gqa_decode's autotune key is
next_power_of_2(B * max_seq_len), so warm up with the exact B and max_seq_len you capture with
(warmup_paged_gqa_decode / capture_cuda_graph below).

NOTE: no `from __future__ import annotations` here: custom_op infers the schema from the real
annotation objects. typing.Optional keeps this Python 3.9 compatible.
"""
from typing import Callable, Dict, Optional, Tuple

import torch

from triton_kernels import gqa_decode as _gqa
from triton_kernels import paged_gqa_decode as _paged
from triton_kernels import paged_kv as _pkv


def _ext():
    try:
        from sparse_kv import _C
    except ImportError as e:
        raise RuntimeError("sparse_kv._C is not built: python setup.py build_ext --inplace") from e
    return _C


# --------------------------------------------------------------------------- CUDA extension ops

@torch.library.custom_op("sparse_kv::fused_gqa_v4", mutates_args=())
def _fused_gqa_v4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  scale: float) -> torch.Tensor:
    return _ext().fused_gqa_v4(q, k, v, scale)


@_fused_gqa_v4.register_fake
def _(q, k, v, scale):
    return q.new_empty(q.shape)


@torch.library.custom_op("sparse_kv::fused_gqa_v4_fp8", mutates_args=())
def _fused_gqa_v4_fp8(q: torch.Tensor, kq: torch.Tensor, vq: torch.Tensor,
                      k_scale: torch.Tensor, v_scale: torch.Tensor, scale: float) -> torch.Tensor:
    return _ext().fused_gqa_v4_fp8(q, kq, vq, k_scale, v_scale, scale)


@_fused_gqa_v4_fp8.register_fake
def _(q, kq, vq, k_scale, v_scale, scale):
    return q.new_empty(q.shape)


@torch.library.custom_op("sparse_kv::fused_gqa_tc", mutates_args=())
def _fused_gqa_tc(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  scale: float, causal: bool) -> torch.Tensor:
    return _ext().fused_gqa_tc(q, k, v, scale, causal)


@_fused_gqa_tc.register_fake
def _(q, k, v, scale, causal):
    return q.new_empty(q.shape)


@torch.library.custom_op("sparse_kv::fused_gqa_tc_fp8", mutates_args=())
def _fused_gqa_tc_fp8(q: torch.Tensor, kq: torch.Tensor, vq: torch.Tensor,
                      k_scale: torch.Tensor, v_scale: torch.Tensor,
                      scale: float, causal: bool) -> torch.Tensor:
    return _ext().fused_gqa_tc_fp8(q, kq, vq, k_scale, v_scale, scale, causal)


@_fused_gqa_tc_fp8.register_fake
def _(q, kq, vq, k_scale, v_scale, scale, causal):
    return q.new_empty(q.shape)


# --------------------------------------------------------------------------------- Triton ops

@torch.library.custom_op("sparse_kv::triton_gqa_decode", mutates_args=())
def _triton_gqa_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                       k_scale: Optional[torch.Tensor], v_scale: Optional[torch.Tensor],
                       num_splits: int) -> torch.Tensor:
    return _gqa.gqa_decode(q, k, v, scale=scale, k_scale=k_scale, v_scale=v_scale,
                           num_splits=num_splits or None)


@_triton_gqa_decode.register_fake
def _(q, k, v, scale, k_scale, v_scale, num_splits):
    return q.new_empty(q.shape)


@torch.library.custom_op("sparse_kv::paged_gqa_decode", mutates_args=())
def _paged_gqa_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                      block_table: torch.Tensor, seq_lens: torch.Tensor, scale: float,
                      k_scale: Optional[torch.Tensor], v_scale: Optional[torch.Tensor],
                      max_seq_len: int, num_splits: int) -> torch.Tensor:
    return _paged.paged_gqa_decode(q, k_cache, v_cache, block_table, seq_lens, scale=scale,
                                   k_scale=k_scale, v_scale=v_scale, max_seq_len=max_seq_len,
                                   num_splits=num_splits or None)


@_paged_gqa_decode.register_fake
def _(q, k_cache, v_cache, block_table, seq_lens, scale, k_scale, v_scale, max_seq_len,
      num_splits):
    return q.new_empty(q.shape)


# Appends mutate the pool in place. Under torch.compile they are functionalized; inductor's
# reinplacing pass turns them back into in-place calls (other backends may clone the pool).
@torch.library.custom_op("sparse_kv::reshape_and_cache", mutates_args=("k_cache", "v_cache"))
def _reshape_and_cache(k_new: torch.Tensor, v_new: torch.Tensor, k_cache: torch.Tensor,
                       v_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    _pkv.reshape_and_cache(k_new, v_new, k_cache, v_cache, slot_mapping)


@_reshape_and_cache.register_fake
def _(k_new, v_new, k_cache, v_cache, slot_mapping):
    return None


@torch.library.custom_op("sparse_kv::reshape_and_cache_fp8",
                         mutates_args=("k_cache", "v_cache", "k_scale", "v_scale"))
def _reshape_and_cache_fp8(k_new: torch.Tensor, v_new: torch.Tensor, k_cache: torch.Tensor,
                           v_cache: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor,
                           slot_mapping: torch.Tensor) -> None:
    _pkv.reshape_and_cache_fp8(k_new, v_new, k_cache, v_cache, k_scale, v_scale, slot_mapping)


@_reshape_and_cache_fp8.register_fake
def _(k_new, v_new, k_cache, v_cache, k_scale, v_scale, slot_mapping):
    return None


# ------------------------------------------------------------------------- Python entry points

def _scale(q, scale):
    return q.shape[-1] ** -0.5 if scale is None else float(scale)


def fused_gqa_v4(q, k, v, scale=None):
    return torch.ops.sparse_kv.fused_gqa_v4(q, k, v, _scale(q, scale))


def fused_gqa_v4_fp8(q, kq, vq, k_scale, v_scale, scale=None):
    return torch.ops.sparse_kv.fused_gqa_v4_fp8(q, kq, vq, k_scale, v_scale, _scale(q, scale))


def fused_gqa_tc(q, k, v, scale=None, causal=False):
    return torch.ops.sparse_kv.fused_gqa_tc(q, k, v, _scale(q, scale), bool(causal))


def fused_gqa_tc_fp8(q, kq, vq, k_scale, v_scale, scale=None, causal=False):
    return torch.ops.sparse_kv.fused_gqa_tc_fp8(
        q, kq, vq, k_scale, v_scale, _scale(q, scale), bool(causal))


def triton_gqa_decode(q, k, v, scale=None, k_scale=None, v_scale=None, num_splits=None):
    return torch.ops.sparse_kv.triton_gqa_decode(q, k, v, _scale(q, scale), k_scale, v_scale,
                                                 num_splits or 0)


def paged_gqa_decode(q, k_cache, v_cache, block_table, seq_lens, max_seq_len, scale=None,
                     k_scale=None, v_scale=None, num_splits=None):
    """max_seq_len is required (>= every seq_lens entry, fixed for a captured graph)."""
    return torch.ops.sparse_kv.paged_gqa_decode(q, k_cache, v_cache, block_table, seq_lens,
                                                _scale(q, scale), k_scale, v_scale,
                                                int(max_seq_len), num_splits or 0)


def reshape_and_cache(k_new, v_new, k_cache, v_cache, slot_mapping, k_scale=None, v_scale=None):
    if k_scale is None:
        torch.ops.sparse_kv.reshape_and_cache(k_new, v_new, k_cache, v_cache, slot_mapping)
    else:
        torch.ops.sparse_kv.reshape_and_cache_fp8(k_new, v_new, k_cache, v_cache, k_scale,
                                                  v_scale, slot_mapping)


# ------------------------------------------------------------------------ CUDA-graph helpers

def autotune_cache_sizes() -> Dict[str, int]:
    """Number of autotuned keys per Triton decode kernel. Constant across a capture means no
    autotuning (benchmarking + synchronize) happened inside it."""
    return {"paged_split": len(_paged._paged_split_kernel.cache),
            "decode_split": len(_gqa._decode_split_kernel.cache)}


def _side_stream_run(fn: Callable[[], object], iters: int):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(iters):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()


def warmup_paged_gqa_decode(q, k_cache, v_cache, block_table, seq_lens, max_seq_len,
                            scale=None, k_scale=None, v_scale=None, num_splits=None, iters=2):
    """JIT-compile and autotune paged_gqa_decode for this exact (B, max_seq_len) bucket, so a
    later capture with the same shapes and max_seq_len replays a cached config."""
    _side_stream_run(lambda: paged_gqa_decode(q, k_cache, v_cache, block_table, seq_lens,
                                              max_seq_len, scale, k_scale, v_scale, num_splits),
                     iters)


def capture_cuda_graph(fn: Callable[[], object], warmup_iters: int = 3,
                       pool=None) -> Tuple[torch.cuda.CUDAGraph, object]:
    """Warm fn up on a side stream (Triton JIT + autotune, cuBLAS handles, allocator), then
    capture one call. fn must read and write only static buffers. Returns (graph, output)."""
    _side_stream_run(fn, warmup_iters)
    before = autotune_cache_sizes()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        out = fn()
    after = autotune_cache_sizes()
    if after != before:
        raise RuntimeError("Triton autotuned during CUDA graph capture ({} -> {}); warm up with "
                           "the exact batch and max_seq_len first".format(before, after))
    return graph, out
