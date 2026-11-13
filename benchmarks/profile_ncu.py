"""
Single-launch driver for Nsight Compute (`ncu`) and SASS inspection (WP-5).

profile_kernel_vs_sdpa.py runs ~226 launches per context, which makes `ncu`
crawl. This runs ONE shape, warms up, then brackets exactly one call with
cudaProfilerStart/Stop. Bracketing (rather than `ncu -s N`) matters for Triton:
its first call autotunes by launching the kernel dozens of times, which would
shift any launch-skip count onto an arbitrary config.

One command profiles the CUDA and Triton decode kernels at one context and
prints the WP-5 Roofline & Stall counters:

    python benchmarks/profile_ncu.py --ncu --ncu-csv --ctx 32768

Inspect compiled CUDA SASS instruction counts (HMMA / LDGSTS / LDG.E.128):

    python benchmarks/profile_ncu.py --sass

A single implementation, for a custom ncu command line:

    ncu --profile-from-start off -k regex:gqa_decode_v4_kernel -c 1 --section SpeedOfLight \\
        python benchmarks/profile_ncu.py --impl v4 --ctx 64000

On WSL2 the counters are blocked until Windows allows them: NVIDIA Control Panel
> Desktop > Enable Developer Settings, then Developer > Manage GPU Performance
Counters > Allow access to all users.

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
import csv
import glob
import io
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HQ, HKV, D = 32, 8, 128
PAGED_BLOCK_SIZE = 64

# ncu kernel-name regex for each implementation.
KERNELS = {
    "v3": "gqa_decode_splitkv_kernel",
    "v4": "gqa_decode_v4_kernel",
    "v4_fp8": "gqa_decode_v4_fp8_kernel",
    "tc": "gqa_decode_tc_kernel",
    "tc_fp8": "gqa_decode_tc_kernel",
    "triton": "_decode_split_kernel",
    "triton_fp8": "_decode_split_kernel",
    "paged_triton": "_paged_split_kernel",
    "paged_fp8": "_paged_split_kernel",
    "quest": "_quest_split_kernel",
    "quest_fp8": "_quest_split_kernel",
    "flashinfer": "Batch(Decode|Prefill)WithPagedKVCacheKernel",
    "sdpa": "flash_fwd|fmha|attention",
}

IMPL_CHOICES = [
    "v3",
    "v4",
    "v4_fp8",
    "tc",
    "tc_fp8",
    "triton",
    "triton_fp8",
    "paged_triton",
    "paged_fp8",
    "quest",
    "quest_fp8",
    "flashinfer",
    "sdpa",
]

DEFAULT_NCU_IMPLS = "v4,tc,tc_fp8,triton,triton_fp8,paged_triton"

METRICS = [
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "sm__inst_issued.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_instruction_executed.avg",
    "smsp__warp_issue_stalled_mio_throttle_per_instruction_executed.avg",
    "lts__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


def _build_paged_inputs(k, v, block_size=PAGED_BLOCK_SIZE, fp8=False, with_minmax=False):
    import torch
    from triton_kernels.paged_kv import (
        allocate_page_minmax,
        allocate_paged_kv_cache,
        reshape_and_cache,
        reshape_and_cache_fp8,
    )

    batch, ctx, hkv, d = k.shape
    num_blocks_per_seq = (ctx + block_size - 1) // block_size
    total_blocks = batch * num_blocks_per_seq
    bt = (
        torch.randperm(total_blocks, device=k.device, dtype=torch.int32)
        .view(batch, num_blocks_per_seq)
        .contiguous()
    )
    sl = torch.full((batch,), ctx, dtype=torch.int32, device=k.device)
    pos = torch.arange(ctx, device=k.device, dtype=torch.int64)
    slot_mapping = (
        (bt[:, pos // block_size].long() * block_size + (pos % block_size))
        .reshape(-1)
        .to(torch.int32)
    )

    kc, vc, ks, vs = allocate_paged_kv_cache(
        total_blocks, block_size, hkv, d, fp8=fp8, device=k.device
    )
    k_min, k_max = (
        allocate_page_minmax(total_blocks, hkv, d, device=k.device)
        if with_minmax
        else (None, None)
    )
    k_flat = k.reshape(batch * ctx, hkv, d)
    v_flat = v.reshape(batch * ctx, hkv, d)
    if fp8:
        reshape_and_cache_fp8(
            k_flat, v_flat, kc, vc, ks, vs, slot_mapping, k_min=k_min, k_max=k_max
        )
    else:
        reshape_and_cache(
            k_flat, v_flat, kc, vc, slot_mapping, k_min=k_min, k_max=k_max
        )
    return kc, vc, ks, vs, bt, sl, k_min, k_max, num_blocks_per_seq


def make_call(impl, q, k, v, quest_budget=0.25):
    import torch
    import torch.nn.functional as F

    scale = D ** -0.5
    ctx = k.shape[1]
    batch = q.shape[0]

    if impl == "sdpa":
        qs = q.unsqueeze(2)
        ks = k.transpose(1, 2).contiguous()
        vs = v.transpose(1, 2).contiguous()
        return lambda: F.scaled_dot_product_attention(qs, ks, vs, scale=scale, enable_gqa=True)

    if impl == "triton":
        from triton_kernels.gqa_decode import gqa_decode

        return lambda: gqa_decode(q, k, v, scale=scale)

    if impl == "triton_fp8":
        from triton_kernels.gqa_decode import gqa_decode, quantize_kv_fp8

        kq, vq, ks, vs = quantize_kv_fp8(k, v)
        return lambda: gqa_decode(q, kq, vq, scale=scale, k_scale=ks, v_scale=vs)

    if impl in ("paged_triton", "paged_fp8"):
        from triton_kernels.paged_gqa_decode import paged_gqa_decode

        fp8 = impl == "paged_fp8"
        kc, vc, ks, vs, bt, sl, _, _, _ = _build_paged_inputs(k, v, fp8=fp8, with_minmax=False)
        return lambda: paged_gqa_decode(
            q, kc, vc, bt, sl, scale=scale, k_scale=ks, v_scale=vs, max_seq_len=ctx
        )

    if impl in ("quest", "quest_fp8"):
        from triton_kernels.quest_sparse_decode import pages_for_budget, quest_sparse_decode

        fp8 = impl == "quest_fp8"
        kc, vc, ks, vs, bt, sl, k_min, k_max, num_blocks_per_seq = _build_paged_inputs(
            k, v, fp8=fp8, with_minmax=True
        )
        recent_pages = 1 if num_blocks_per_seq >= 2 else 0
        top_k_pages = pages_for_budget(
            quest_budget, ctx, PAGED_BLOCK_SIZE, recent_pages=recent_pages
        )
        return lambda: quest_sparse_decode(
            q,
            kc,
            vc,
            bt,
            sl,
            k_min,
            k_max,
            top_k_pages,
            recent_pages=recent_pages,
            scale=scale,
            k_scale=ks,
            v_scale=vs,
            max_seq_len=ctx,
        )

    if impl == "flashinfer":
        try:
            import flashinfer
        except ImportError as exc:
            raise RuntimeError(
                "flashinfer is not installed; install `flashinfer-python` to profile --impl flashinfer"
            ) from exc

        kc, vc, _, _, bt, _, _, _, num_blocks_per_seq = _build_paged_inputs(
            k, v, fp8=False, with_minmax=False
        )
        indptr = torch.arange(
            0,
            (batch + 1) * num_blocks_per_seq,
            num_blocks_per_seq,
            dtype=torch.int32,
            device=q.device,
        )
        indices = bt.reshape(-1).contiguous()
        last_page_len = torch.full(
            (batch,), (ctx - 1) % PAGED_BLOCK_SIZE + 1, dtype=torch.int32, device=q.device
        )
        ws = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            HQ,
            HKV,
            D,
            PAGED_BLOCK_SIZE,
            pos_encoding_mode="NONE",
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )
        return lambda: wrapper.run(q, (kc, vc))

    import sparse_kv._C as _C

    if impl in ("v4_fp8", "tc_fp8"):
        from triton_kernels.gqa_decode import quantize_kv_fp8

        kq, vq, ks, vs = quantize_kv_fp8(k, v)
        if impl == "v4_fp8":
            return lambda: _C.fused_gqa_v4_fp8(q, kq, vq, ks, vs, scale)
        return lambda: _C.fused_gqa_tc_fp8(q, kq, vq, ks, vs, scale)

    fn = {"v3": _C.fused_gqa, "v4": _C.fused_gqa_v4, "tc": _C.fused_gqa_tc}[impl]
    return lambda: fn(q, k, v, scale)


def _parse_float(val: str) -> Optional[float]:
    cleaned = val.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_ncu_csv(csv_text: str) -> Tuple[str, Dict[str, Tuple[Optional[float], str]]]:
    """Parse `ncu --csv` output into (kernel_name, {metric_name: (value, unit)})."""
    lines = csv_text.splitlines()
    header = None  # type: Optional[List[str]]
    data_lines = []  # type: List[str]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if header is None:
            if "Metric Name" in stripped and "Metric Value" in stripped:
                parsed = next(csv.reader([stripped]))
                if "Metric Name" in parsed and "Metric Value" in parsed:
                    header = parsed
            continue
        if stripped.startswith("==PROF==") or stripped.startswith("==ERROR=="):
            continue
        row = next(csv.reader([stripped]))
        if len(row) == len(header):
            data_lines.append(stripped)

    if header is None or not data_lines:
        return "", {}

    reader = csv.DictReader(io.StringIO("\n".join(data_lines)), fieldnames=header)
    kernel_name = ""
    metrics = {}  # type: Dict[str, Tuple[Optional[float], str]]
    for row in reader:
        if not kernel_name:
            kernel_name = (row.get("Kernel Name") or "").strip()
        mname = (row.get("Metric Name") or "").strip()
        munit = (row.get("Metric Unit") or "").strip()
        mval = _parse_float(row.get("Metric Value") or "")
        if mname:
            metrics[mname] = (mval, munit)
    return kernel_name, metrics


def _short_kernel_name(name: str) -> str:
    if not name:
        return "-"
    # Trim argument list from demangled C++ signatures.
    base = name.split("(", 1)[0].strip()
    if "::" in base:
        base = base.split("::")[-1]
    if base.startswith("void "):
        base = base[5:].strip()
    return base


def _to_us(val: Optional[float], unit: str) -> str:
    if val is None:
        return "-"
    u = unit.lower()
    if u in ("ns", "nsecond"):
        us = val / 1e3
    elif u in ("ms", "msecond"):
        us = val * 1e3
    elif u in ("s", "second"):
        us = val * 1e6
    else:
        us = val
    return "{:.2f}".format(us)


def _to_mb(val: Optional[float], unit: str) -> str:
    if val is None:
        return "-"
    u = unit.lower()
    if u in ("byte", "bytes", "b"):
        mb = val / 1e6
    elif u in ("kbyte", "kb"):
        mb = val / 1e3
    elif u in ("gbyte", "gb"):
        mb = val * 1e3
    else:
        mb = val
    return "{:.2f}".format(mb)


def _fmt_num(val: Optional[float], fmt: str = "{:.2f}") -> str:
    if val is None:
        return "-"
    return fmt.format(val)


def format_ncu_markdown(
    rows: List[Tuple[str, str, Dict[str, Tuple[Optional[float], str]], Optional[str]]],
    batch: int,
    ctx: int,
) -> str:
    """Format parsed ncu CSV results into a Markdown Roofline & Stall summary table."""
    lines = [
        "# Nsight Compute Roofline & Stall Summary (batch={}, ctx={})".format(batch, ctx),
        "",
        "| Impl | Kernel | Duration (us) | DRAM % | DRAM Read (MB) | Inst Issued % | "
        "HMMA % | Long Scoreboard | MIO Throttle | L2 Hit % | Warps Active % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for impl, kernel_name, metrics, err in rows:
        if err is not None:
            lines.append(
                "| {} | {} | - | - | - | - | - | - | - | - | - |".format(
                    impl, err.replace("|", "/")
                )
            )
            continue

        def get(mname: str) -> Tuple[Optional[float], str]:
            return metrics.get(mname, (None, ""))

        dur_v, dur_u = get("gpu__time_duration.sum")
        dram_pct, _ = get("dram__throughput.avg.pct_of_peak_sustained_elapsed")
        dram_b, dram_bu = get("dram__bytes_read.sum")
        inst_pct, _ = get("sm__inst_issued.avg.pct_of_peak_sustained_active")
        hmma_pct, _ = get("sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active")
        long_sb, _ = get("smsp__warp_issue_stalled_long_scoreboard_per_instruction_executed.avg")
        mio_th, _ = get("smsp__warp_issue_stalled_mio_throttle_per_instruction_executed.avg")
        l2_hit, _ = get("lts__t_sector_hit_rate.pct")
        warps, _ = get("sm__warps_active.avg.pct_of_peak_sustained_active")

        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                impl,
                _short_kernel_name(kernel_name),
                _to_us(dur_v, dur_u),
                _fmt_num(dram_pct),
                _to_mb(dram_b, dram_bu),
                _fmt_num(inst_pct),
                _fmt_num(hmma_pct),
                _fmt_num(long_sb, "{:.3f}"),
                _fmt_num(mio_th, "{:.3f}"),
                _fmt_num(l2_hit),
                _fmt_num(warps),
            )
        )
    return "\n".join(lines) + "\n"


def _write_output(path: str, content: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def run_ncu(args):
    impls = [x.strip() for x in args.ncu_impls.split(",") if x.strip()]
    invalid = [x for x in impls if x not in KERNELS]
    if invalid:
        raise SystemExit(
            "Unknown --ncu-impls entries: {}. Valid choices: {}".format(
                ", ".join(invalid), ", ".join(IMPL_CHOICES)
            )
        )

    raw_chunks = []  # type: List[str]
    parsed_rows = []  # type: List[Tuple[str, str, Dict[str, Tuple[Optional[float], str]], Optional[str]]]
    failures = 0

    for impl in impls:
        header = "== {} ({}), ctx={}".format(impl, KERNELS[impl], args.ctx)
        print(header, flush=True)
        cmd = [
            args.ncu_bin,
            "--profile-from-start",
            "off",
            "-k",
            "regex:" + KERNELS[impl],
            "-c",
            "1",
            "--metrics",
            ",".join(METRICS),
        ]
        if args.ncu_csv:
            cmd.append("--csv")
        cmd.extend(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--impl",
                impl,
                "--ctx",
                str(args.ctx),
                "--batch",
                str(args.batch),
                "--warmup",
                str(args.warmup),
                "--quest-budget",
                str(args.quest_budget),
            ]
        )

        if args.ncu_csv or args.ncu_out:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            combined = proc.stdout
            if proc.stderr:
                combined = combined + ("\n" if combined else "") + proc.stderr
            print(combined, end="" if combined.endswith("\n") else "\n", flush=True)
            raw_chunks.append(header + "\n" + combined.rstrip() + "\n")
            if proc.returncode != 0:
                failures += 1
                parsed_rows.append((impl, "", {}, "error (exit {})".format(proc.returncode)))
            elif args.ncu_csv:
                kname, metrics = parse_ncu_csv(combined)
                err = None if metrics else "no CSV rows"
                parsed_rows.append((impl, kname, metrics, err))
        else:
            subprocess.run(cmd, check=True)

    if args.ncu_csv:
        md = format_ncu_markdown(parsed_rows, args.batch, args.ctx)
        print("\n" + md, end="", flush=True)
        if args.ncu_out:
            full_doc = (
                md
                + "\n## Raw `ncu --csv` Output\n\n```text\n"
                + "\n".join(raw_chunks).rstrip()
                + "\n```\n"
            )
            _write_output(args.ncu_out, full_doc)
            print("saved -> {}".format(args.ncu_out))
    elif args.ncu_out:
        doc = (
            "# Nsight Compute Raw Output (batch={}, ctx={})\n\n```text\n".format(
                args.batch, args.ctx
            )
            + "\n".join(raw_chunks).rstrip()
            + "\n```\n"
        )
        _write_output(args.ncu_out, doc)
        print("saved -> {}".format(args.ncu_out))

    if failures == len(impls) and impls:
        raise SystemExit(1)


# ---------------------------------------------------------------------------- SASS inspection

_ARCH_RE = re.compile(r"^\s*code\s+for\s+(sm_\w+)", re.IGNORECASE)
_FUNC_RE = re.compile(r"^\s*Function\s*:\s*(.+?)\s*$")
_DECODE_FILTER_RE = re.compile(r"(gqa_decode_\w+|gqa_combine\w*)")
_MANGLED_NAME_RE = re.compile(r"(\d+)(gqa_(?:decode_|combine)\w*)")
_TMPL_ARGS_RE = re.compile(r"^I((?:Li\d+E|Lb[01]E)+)E")
_INST_RE = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?\w+\s+)?([A-Za-z][A-Za-z0-9_.]*)")


def _clean_sass_func_name(raw_name: str) -> str:
    m_len = _MANGLED_NAME_RE.search(raw_name)
    if m_len:
        digits = m_len.group(1)
        rest = m_len.group(2)
        for width in (3, 2, 1):
            if len(digits) < width:
                continue
            length = int(digits[-width:])
            if length <= 0 or length > len(rest):
                continue
            base = rest[:length]
            if not re.fullmatch(r"[a-z0-9_]+", base):
                continue
            tail = raw_name[m_len.start(2) + length :]
            tmpl_m = _TMPL_ARGS_RE.match(tail)
            if tmpl_m:
                raw_args = tmpl_m.group(1)
                args = []
                for tok in re.findall(r"Li(\d+)E|Lb([01])E", raw_args):
                    if tok[0]:
                        args.append(tok[0])
                    else:
                        args.append("true" if tok[1] == "1" else "false")
                return "{}<{}>".format(base, ",".join(args))
            return base

    # Demangled or plain name: keep from gqa_decode_/gqa_combine up to parameter list.
    m_plain = _DECODE_FILTER_RE.search(raw_name)
    if not m_plain:
        return raw_name.strip()
    sub = raw_name[m_plain.start() :]
    return sub.split("(", 1)[0].strip()


def summarize_sass(sass_text: str) -> str:
    """Summarize per-function SASS instruction counts for decode kernels into Markdown.

    Pure helper over `cuobjdump -sass` text so it can be unit-tested without CUDA.
    """
    entries = []  # type: List[Tuple[str, str, Dict[str, int]]]
    current_arch = ""
    current_func = None  # type: Optional[str]
    current_counts = None  # type: Optional[Dict[str, int]]

    def _new_counts() -> Dict[str, int]:
        return {
            "HMMA": 0,
            "LDGSTS": 0,
            "LDG.E.128": 0,
            "LDG": 0,
            "LDS": 0,
            "LDSM": 0,
            "FFMA/HFMA2": 0,
            "Total": 0,
        }

    def _flush() -> None:
        if current_func is not None and current_counts is not None:
            if _DECODE_FILTER_RE.search(current_func):
                entries.append((current_arch, _clean_sass_func_name(current_func), current_counts))

    for line in sass_text.splitlines():
        arch_m = _ARCH_RE.match(line)
        if arch_m:
            _flush()
            current_arch = arch_m.group(1)
            current_func = None
            current_counts = None
            continue

        func_m = _FUNC_RE.match(line)
        if func_m:
            _flush()
            current_func = func_m.group(1)
            current_counts = _new_counts()
            continue

        if current_counts is None:
            continue

        inst_m = _INST_RE.match(line)
        if not inst_m:
            continue

        op = inst_m.group(1).upper()
        current_counts["Total"] += 1
        if op == "HMMA" or op.startswith("HMMA."):
            current_counts["HMMA"] += 1
        elif op == "LDGSTS" or op.startswith("LDGSTS."):
            current_counts["LDGSTS"] += 1
        elif op == "LDSM" or op.startswith("LDSM."):
            current_counts["LDSM"] += 1
        elif op == "LDS" or op.startswith("LDS."):
            current_counts["LDS"] += 1
        elif op == "LDG" or op.startswith("LDG."):
            current_counts["LDG"] += 1
            if op == "LDG.E.128" or op.startswith("LDG.E.128."):
                current_counts["LDG.E.128"] += 1
        elif (
            op == "FFMA"
            or op.startswith("FFMA.")
            or op == "HFMA2"
            or op.startswith("HFMA2.")
        ):
            current_counts["FFMA/HFMA2"] += 1

    _flush()

    if not entries:
        return "No decode kernels (`gqa_decode_*`, `gqa_combine*`) found in SASS output.\n"

    distinct_archs = {arch for arch, _, _ in entries if arch}
    show_arch = len(distinct_archs) > 1
    arch_note = " ({})".format(next(iter(distinct_archs))) if len(distinct_archs) == 1 else ""

    lines = [
        "# Decode Kernel SASS Instruction Summary" + arch_note,
        "",
        "| Kernel | HMMA | LDGSTS | LDG.E.128 | LDG | LDS | LDSM | FFMA/HFMA2 | Total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arch, name, counts in entries:
        label = "{} [{}]".format(name, arch) if (show_arch and arch) else name
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                label,
                counts["HMMA"],
                counts["LDGSTS"],
                counts["LDG.E.128"],
                counts["LDG"],
                counts["LDS"],
                counts["LDSM"],
                counts["FFMA/HFMA2"],
                counts["Total"],
            )
        )
    return "\n".join(lines) + "\n"


def _find_extension_so() -> str:
    try:
        import sparse_kv._C as _C

        path = getattr(_C, "__file__", None)
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    matches = sorted(glob.glob(os.path.join(ROOT, "sparse_kv", "_C*.so")))
    if matches:
        return matches[0]
    raise SystemExit(
        "Could not find sparse_kv._C extension (.so). Build it with: python setup.py build_ext --inplace"
    )


def run_sass(args):
    so_path = _find_extension_so()
    try:
        proc = subprocess.run(
            ["cuobjdump", "-sass", so_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "cuobjdump not found in PATH; install/load the CUDA toolkit to use --sass"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "cuobjdump -sass {} failed (exit {}):\n{}".format(
                so_path, exc.returncode, exc.stderr
            )
        ) from exc

    summary = summarize_sass(proc.stdout)
    print(summary, end="")
    if args.sass_out:
        _write_output(args.sass_out, summary)
        print("saved -> {}".format(args.sass_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3, help="calls before the profiled one")
    ap.add_argument("--impl", choices=IMPL_CHOICES, default="v4")
    ap.add_argument(
        "--quest-budget",
        type=float,
        default=0.25,
        help="fractional page budget for quest/quest_fp8 (default: 0.25)",
    )
    ap.add_argument(
        "--ncu",
        action="store_true",
        help="run ncu on --ncu-impls and print their Roofline & Stall counters",
    )
    ap.add_argument("--ncu-bin", default="ncu")
    ap.add_argument(
        "--ncu-impls",
        default=DEFAULT_NCU_IMPLS,
        help="comma-separated implementations for --ncu mode",
    )
    ap.add_argument(
        "--ncu-out",
        default=None,
        help="optional file path to write Markdown + raw ncu output",
    )
    ap.add_argument(
        "--ncu-csv",
        action="store_true",
        help="use `ncu --csv` internally and parse metrics into a Markdown summary table",
    )
    ap.add_argument(
        "--sass",
        action="store_true",
        help="run `cuobjdump -sass` on sparse_kv._C and summarize decode kernel instructions",
    )
    ap.add_argument(
        "--sass-out",
        default=None,
        help="optional file path to save the --sass Markdown summary",
    )
    args = ap.parse_args()
    if args.sass:
        return run_sass(args)
    if args.ncu:
        return run_ncu(args)

    import torch

    torch.manual_seed(0)
    q = torch.randn(args.batch, HQ, D, device="cuda", dtype=torch.float16)
    k = torch.randn(args.batch, args.ctx, HKV, D, device="cuda", dtype=torch.float16)
    v = torch.randn(args.batch, args.ctx, HKV, D, device="cuda", dtype=torch.float16)
    fn = make_call(args.impl, q, k, v, quest_budget=args.quest_budget)

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    elem_bytes = 1 if args.impl.endswith("_fp8") else 2
    print(
        "profiled {} | batch={} ctx={} | unique KV {:.3f} GB".format(
            args.impl, args.batch, args.ctx, 2 * k.numel() * elem_bytes / 1e9
        )
    )


if __name__ == "__main__":
    main()

