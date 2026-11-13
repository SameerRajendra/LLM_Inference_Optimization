"""End-to-end Nsight Systems (nsys) & CUDA-event decode profiler (WP-5 Task 2).

Annotates every stage of a real decode step (reusing `load_model`, `Model`,
`PrefillCache`, `DecodeState`, and `rms_norm` from `benchmarks.bench_e2e_serving`)
with exact `torch.cuda.nvtx.range_push` / `range_pop` markers:

  Decode_Step (outer)
    Embedding
    RMSNorm            (pre-attn, Qwen3 q/k norm if present, post-attn, final norm)
    QKV_Proj
    RoPE_KV_Append
    Quest_Page_Select  (quest_fp16 / quest_fp8 layers >= dense_prefix_layers)
    Paged_Attention
    O_Proj
    MLP_SwiGLU
    LM_Head

Three execution modes:
  1. Single-config worker mode (for `nsys profile`):
     nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \\
       --capture-range-end=stop -o profiles/trace \\
       python benchmarks/profile_e2e_nsys.py --method paged_fp16 --batch 16 --ctx 4096
  2. Lightweight CUDA-events breakdown mode (when `nsys` is unavailable):
     python benchmarks/profile_e2e_nsys.py --cuda-events --method quest_fp8 --batch 16 --ctx 4096
  3. Sweep driver mode (`--nsys-sweep`):
     Runs `nsys profile` -> `nsys stats --report nvtx_sum --format csv` across
     `--batches` x `--contexts` and writes `profiles/INFERENCE_PROFILE.md`.

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import io
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

METHODS = ("paged_fp16", "paged_fp8", "quest_fp16", "quest_fp8")

STAGE_ORDER = (
    "Embedding",
    "RMSNorm",
    "QKV_Proj",
    "RoPE_KV_Append",
    "Quest_Page_Select",
    "Paged_Attention",
    "O_Proj",
    "MLP_SwiGLU",
    "LM_Head",
)
OUTER_STAGE = "Decode_Step"
ALL_KNOWN_STAGES = set(STAGE_ORDER) | {OUTER_STAGE}


# --------------------------------------------------------------------------- pure CSV & Markdown

def _unit_to_ms(col_header: str) -> float:
    m = re.search(r"\((ns|us|ms|s)\)", col_header, re.IGNORECASE)
    if not m:
        return 1e-6
    unit = m.group(1).lower()
    if unit == "ns":
        return 1e-6
    if unit == "us":
        return 1e-3
    if unit == "ms":
        return 1.0
    if unit == "s":
        return 1e3
    return 1e-6


def _base_col_name(col_header: str) -> str:
    s = re.sub(r"\s*\([^)]*\)\s*", " ", col_header).strip().lower()
    return re.sub(r"\s+", " ", s)


def _parse_float(val: str, default: Optional[float] = 0.0) -> Optional[float]:
    s = val.strip().replace(",", "")
    if not s or s in ("-", "N/A", "n/a", "nan", "NaN"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _parse_int(val: str, default: int = 0) -> int:
    f = _parse_float(val, None)
    if f is None:
        return default
    return int(round(f))


def _clean_range_name(raw: str) -> str:
    name = raw.strip()
    if name.startswith(":"):
        name = name[1:]
    elif ":" in name:
        _, tail = name.split(":", 1)
        if tail in ALL_KNOWN_STAGES:
            name = tail
    return name


def parse_nvtx_csv(csv_text: str) -> List[Dict[str, Any]]:
    """Parse `nsys stats --report nvtx_sum --format csv` output into normalized records.

    Skips any leading `nsys` status/banner lines, converts time columns to milliseconds
    based on header units (`ns`, `us`, `ms`, `s`), and strips default-domain `:` prefixes
    from NVTX range names.
    """
    if not csv_text or not csv_text.strip():
        return []

    lines = csv_text.splitlines()
    header_idx = None
    headers = []  # type: List[str]
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or "," not in stripped:
            continue
        parsed_row = next(csv.reader([stripped]))
        bases = [_base_col_name(c) for c in parsed_row]
        has_range = any(b in ("range", "nvtx range", "name") for b in bases)
        has_time = any(b in ("total time", "time") for b in bases)
        if has_range and has_time:
            header_idx = i
            headers = parsed_row
            break

    if header_idx is None:
        return []

    col_map = {}  # type: Dict[str, int]
    col_units = {}  # type: Dict[str, float]
    for idx, h in enumerate(headers):
        base = _base_col_name(h)
        if base in ("range", "nvtx range", "name") and "range" not in col_map:
            col_map["range"] = idx
        elif base == "time" and "%" in h and "time_pct" not in col_map:
            col_map["time_pct"] = idx
        elif base == "total time" and "total" not in col_map:
            col_map["total"] = idx
            col_units["total"] = _unit_to_ms(h)
        elif base in ("instances", "count", "calls") and "instances" not in col_map:
            col_map["instances"] = idx
        elif base == "avg" and "avg" not in col_map:
            col_map["avg"] = idx
            col_units["avg"] = _unit_to_ms(h)
        elif base in ("med", "median") and "med" not in col_map:
            col_map["med"] = idx
            col_units["med"] = _unit_to_ms(h)
        elif base == "min" and "min" not in col_map:
            col_map["min"] = idx
            col_units["min"] = _unit_to_ms(h)
        elif base == "max" and "max" not in col_map:
            col_map["max"] = idx
            col_units["max"] = _unit_to_ms(h)
        elif base in ("stddev", "std dev", "std") and "stddev" not in col_map:
            col_map["stddev"] = idx
            col_units["stddev"] = _unit_to_ms(h)
        elif base == "style" and "style" not in col_map:
            col_map["style"] = idx

    if "range" not in col_map or "total" not in col_map:
        return []

    records = []  # type: List[Dict[str, Any]]
    body = "\n".join(lines[header_idx + 1:])
    reader = csv.reader(io.StringIO(body))
    for row in reader:
        if not row or len(row) <= col_map["range"]:
            continue
        if len(row) <= col_map["total"]:
            continue
        raw_range = row[col_map["range"]].strip()
        if not raw_range:
            continue
        total_raw = _parse_float(row[col_map["total"]], None)
        if total_raw is None:
            continue
        total_ms = total_raw * col_units["total"]
        instances = _parse_int(row[col_map["instances"]], 0) if "instances" in col_map else 0
        time_pct = _parse_float(row[col_map["time_pct"]], 0.0) if "time_pct" in col_map else 0.0
        if "avg" in col_map and col_map["avg"] < len(row):
            avg_val = _parse_float(row[col_map["avg"]], None)
            avg_ms = (avg_val * col_units["avg"]) if avg_val is not None else (
                total_ms / max(instances, 1)
            )
        else:
            avg_ms = total_ms / max(instances, 1)

        rec = {
            "range": _clean_range_name(raw_range),
            "time_pct": float(time_pct if time_pct is not None else 0.0),
            "total_ms": float(total_ms),
            "instances": instances,
            "avg_ms": float(avg_ms),
        }  # type: Dict[str, Any]

        for opt_key in ("med", "min", "max", "stddev"):
            if opt_key in col_map and col_map[opt_key] < len(row):
                v = _parse_float(row[col_map[opt_key]], None)
                if v is not None:
                    rec[opt_key + "_ms"] = float(v * col_units[opt_key])
        if "style" in col_map and col_map["style"] < len(row):
            rec["style"] = row[col_map["style"]].strip()
        records.append(rec)

    return records


def _coerce_stages(stages_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(stages_obj, list):
        out = []  # type: List[Dict[str, Any]]
        for item in stages_obj:
            if not isinstance(item, dict):
                continue
            name = item.get("range") or item.get("Range") or item.get("name")
            if not name:
                continue
            out.append({
                "range": _clean_range_name(str(name)),
                "total_ms": float(item.get("total_ms", item.get("ms", 0.0))),
                "instances": int(item.get("instances", item.get("calls", 0))),
            })
        return out
    if isinstance(stages_obj, dict):
        out = []
        for name, val in stages_obj.items():
            if isinstance(val, dict):
                out.append({
                    "range": _clean_range_name(str(name)),
                    "total_ms": float(val.get("total_ms", val.get("ms", 0.0))),
                    "instances": int(val.get("instances", val.get("calls", 0))),
                })
            else:
                out.append({
                    "range": _clean_range_name(str(name)),
                    "total_ms": float(val),
                    "instances": 0,
                })
        return out
    return []


def _normalize_runs(
    records: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not records:
        return []

    first = records[0]
    if "stages" in first or "error" in first or "skipped" in first:
        runs = []
        for rec in records:
            runs.append({
                "method": rec.get("method", metadata.get("method", "unknown")),
                "batch": rec.get("batch", metadata.get("batch")),
                "ctx": rec.get("ctx", metadata.get("ctx")),
                "profile_steps": rec.get("profile_steps", metadata.get("profile_steps")),
                "stages": _coerce_stages(rec.get("stages", [])),
                "error": rec.get("error") or rec.get("skipped"),
            })
        return runs

    # Flat stage records (directly from parse_nvtx_csv, optionally tagged with batch/ctx/method).
    grouped = {}  # type: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]]
    order = []  # type: List[Tuple[Any, Any, Any]]
    steps_by_key = {}  # type: Dict[Tuple[Any, Any, Any], Any]
    for rec in records:
        key = (
            rec.get("method", metadata.get("method", "unknown")),
            rec.get("batch", metadata.get("batch")),
            rec.get("ctx", metadata.get("ctx")),
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
            steps_by_key[key] = rec.get("profile_steps", metadata.get("profile_steps"))
        grouped[key].append(rec)

    runs = []
    for key in order:
        method, batch, ctx = key
        runs.append({
            "method": method,
            "batch": batch,
            "ctx": ctx,
            "profile_steps": steps_by_key[key],
            "stages": _coerce_stages(grouped[key]),
            "error": None,
        })
    return runs


def _summarize_run(run: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    by_stage = {}  # type: Dict[str, Dict[str, Any]]
    for st in run["stages"]:
        name = st["range"]
        if name not in by_stage:
            by_stage[name] = {"total_ms": 0.0, "instances": 0}
        by_stage[name]["total_ms"] += float(st.get("total_ms", 0.0))
        by_stage[name]["instances"] += int(st.get("instances", 0))

    decode_entry = by_stage.get(OUTER_STAGE)
    if decode_entry and decode_entry["instances"] > 0:
        num_steps = decode_entry["instances"]
    else:
        num_steps = int(run.get("profile_steps") or metadata.get("profile_steps") or 1)
    num_steps = max(num_steps, 1)

    child_names = [n for n in by_stage if n != OUTER_STAGE]
    if decode_entry and decode_entry["total_ms"] > 0:
        step_total_ms = decode_entry["total_ms"] / num_steps
    else:
        step_total_ms = sum(by_stage[n]["total_ms"] for n in child_names) / num_steps

    stage_stats = {}  # type: Dict[str, Dict[str, float]]
    attributed_ms = 0.0
    for name in child_names:
        ms_step = by_stage[name]["total_ms"] / num_steps
        pct = (100.0 * ms_step / step_total_ms) if step_total_ms > 0 else 0.0
        calls_step = by_stage[name]["instances"] / num_steps
        stage_stats[name] = {
            "ms_step": ms_step,
            "pct": pct,
            "calls_step": calls_step,
        }
        attributed_ms += ms_step

    other_ms = max(0.0, step_total_ms - attributed_ms)
    if decode_entry and other_ms > 1e-4:
        stage_stats["Other"] = {
            "ms_step": other_ms,
            "pct": (100.0 * other_ms / step_total_ms) if step_total_ms > 0 else 0.0,
            "calls_step": 0.0,
        }

    return {
        "method": run["method"],
        "batch": run["batch"],
        "ctx": run["ctx"],
        "num_steps": num_steps,
        "step_total_ms": step_total_ms,
        "stages": stage_stats,
        "error": run.get("error"),
    }


def _config_label(summary: Dict[str, Any]) -> str:
    parts = []
    if summary["batch"] is not None:
        parts.append("B={}".format(summary["batch"]))
    if summary["ctx"] is not None:
        ctx = int(summary["ctx"])
        parts.append("ctx={}".format("{}K".format(ctx // 1024) if ctx >= 1024 and ctx % 1024 == 0 else ctx))
    if not parts:
        return str(summary["method"])
    return ", ".join(parts)


def format_profile_markdown(
    records: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Format parsed NVTX or CUDA-event profiling records into a Markdown report.

    Produces a metadata header, a cross-configuration comparison matrix (when multiple
    `(batch, ctx)` configurations are present), and per-configuration breakdown tables
    with `ms/step` and `% of step`.
    """
    meta = metadata or {}
    runs = _normalize_runs(records, meta)
    summaries = [_summarize_run(r, meta) for r in runs]

    lines = ["# End-to-End Inference Decode Profile", ""]

    # Metadata block
    meta_rows = []
    if meta.get("model"):
        meta_rows.append(("Model", "`{}`".format(meta["model"])))
    method = meta.get("method") or (summaries[0]["method"] if summaries else None)
    if method and method != "unknown":
        meta_rows.append(("Method", "`{}`".format(method)))
    if meta.get("gpu"):
        meta_rows.append(("GPU", str(meta["gpu"])))
    if meta.get("profile_steps") is not None:
        step_info = "`{}`".format(meta["profile_steps"])
        extras = []
        if meta.get("warmup") is not None:
            extras.append("warmup `{}`".format(meta["warmup"]))
        if meta.get("block_size") is not None:
            extras.append("block size `{}`".format(meta["block_size"]))
        if extras:
            step_info += " ({})".format(", ".join(extras))
        meta_rows.append(("Profile steps", step_info))
    if method and str(method).startswith("quest"):
        q_parts = []
        if meta.get("quest_budget") is not None:
            q_parts.append("budget `{}`".format(meta["quest_budget"]))
        if meta.get("dense_prefix_layers") is not None:
            q_parts.append("dense prefix layers `{}`".format(meta["dense_prefix_layers"]))
        if meta.get("recent_pages") is not None:
            q_parts.append("recent pages `{}`".format(meta["recent_pages"]))
        if q_parts:
            meta_rows.append(("Quest config", ", ".join(q_parts)))
    if meta.get("source"):
        meta_rows.append(("Profiler source", "`{}`".format(meta["source"])))
    if meta.get("utc"):
        meta_rows.append(("Generated (UTC)", "`{}`".format(meta["utc"])))

    for k, v in meta_rows:
        lines.append("- **{}**: {}".format(k, v))
    if meta_rows:
        lines.append("")

    valid_summaries = [s for s in summaries if not s["error"] and (s["stages"] or s["step_total_ms"] > 0)]
    if not valid_summaries:
        lines.append("_No profiling records available._")
        for s in summaries:
            if s["error"]:
                lines.append("- `{}`: {}".format(_config_label(s), s["error"]))
        lines.append("")
        return "\n".join(lines)

    # Determine ordered stage list across all runs
    seen_stages = set()
    for s in valid_summaries:
        seen_stages.update(s["stages"].keys())
    ordered_stages = [st for st in STAGE_ORDER if st in seen_stages]
    extra_stages = sorted(st for st in seen_stages if st not in STAGE_ORDER and st != "Other")
    ordered_stages.extend(extra_stages)
    if "Other" in seen_stages:
        ordered_stages.append("Other")

    # Cross-configuration matrix when more than 1 valid config is present
    if len(valid_summaries) > 1:
        lines.append("## Summary Breakdown (`ms/step` and `% of step`)")
        lines.append("")
        col_headers = ["Stage"] + [_config_label(s) for s in valid_summaries]
        lines.append("| " + " | ".join(col_headers) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(valid_summaries)) + "|")
        for st in ordered_stages:
            row_cells = ["`{}`".format(st)]
            for s in valid_summaries:
                info = s["stages"].get(st)
                if info is None:
                    row_cells.append("-")
                else:
                    row_cells.append("{:.3f} ms ({:.1f}%)".format(info["ms_step"], info["pct"]))
            lines.append("| " + " | ".join(row_cells) + " |")
        total_cells = ["**`{}` (Total)**".format(OUTER_STAGE)]
        for s in valid_summaries:
            total_cells.append("**{:.3f} ms (100.0%)**".format(s["step_total_ms"]))
        lines.append("| " + " | ".join(total_cells) + " |")
        lines.append("")

    # Detailed per-configuration breakdown tables
    lines.append("## Per-Configuration Breakdown")
    lines.append("")
    for s in summaries:
        label = _config_label(s)
        if s["method"] and s["method"] != "unknown" and s["method"] != method:
            label = "{} ({})".format(label, s["method"])
        lines.append("### {}".format(label))
        lines.append("")
        if s["error"]:
            lines.append("_Skipped / failed: {}_".format(s["error"]))
            lines.append("")
            continue
        lines.append("| Stage | ms/step | % of step | Calls/step |")
        lines.append("|---|---:|---:|---:|")
        for st in ordered_stages:
            info = s["stages"].get(st)
            if info is None:
                continue
            calls_str = "-" if st == "Other" else "{:.1f}".format(info["calls_step"])
            lines.append("| `{}` | {:.3f} | {:.1f}% | {} |".format(
                st, info["ms_step"], info["pct"], calls_str
            ))
        lines.append("| **`{}` (Total)** | **{:.3f}** | **100.0%** | **1.0** |".format(
            OUTER_STAGE, s["step_total_ms"]
        ))
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- GPU profiling path

def _import_gpu_deps():
    import torch
    import torch.nn.functional as F
    from benchmarks.bench_e2e_serving import (
        DecodeState,
        Model,
        PrefillCache,
        load_model,
        rms_norm,
    )
    from sparse_kv import ops as kv_ops
    from triton_kernels.paged_kv import (
        allocate_page_minmax,
        apply_rope,
        update_page_minmax,
    )
    from triton_kernels.quest_sparse_decode import (
        paged_sparse_decode,
        pages_for_budget,
        quest_page_scores,
        quest_select_pages,
    )
    return {
        "torch": torch,
        "F": F,
        "DecodeState": DecodeState,
        "Model": Model,
        "PrefillCache": PrefillCache,
        "load_model": load_model,
        "rms_norm": rms_norm,
        "kv_ops": kv_ops,
        "allocate_page_minmax": allocate_page_minmax,
        "apply_rope": apply_rope,
        "update_page_minmax": update_page_minmax,
        "paged_sparse_decode": paged_sparse_decode,
        "pages_for_budget": pages_for_budget,
        "quest_page_scores": quest_page_scores,
        "quest_select_pages": quest_select_pages,
    }


def _make_profile_state(
    deps,
    M,
    method: str,
    batch: int,
    ctx: int,
    gen: int,
    block_size: int,
    device: str,
    quest_budget: float,
    dense_prefix_layers: int,
    recent_pages: int,
    seed: int,
):
    torch = deps["torch"]
    DecodeState = deps["DecodeState"]
    allocate_page_minmax = deps["allocate_page_minmax"]
    update_page_minmax = deps["update_page_minmax"]
    kv_ops = deps["kv_ops"]

    class ProfileDecodeState(DecodeState):
        """Extends `DecodeState` with Quest per-page min/max metadata when needed."""

        def __init__(self):
            base_method = "paged_fp8" if method.endswith("fp8") else "paged_fp16"
            super().__init__(M, base_method, batch, ctx, gen, block_size, device, seed=seed)
            self.profile_method = method
            self.use_quest = method.startswith("quest")
            self.quest_budget = float(quest_budget)
            self.dense_prefix_layers = int(dense_prefix_layers)
            self.recent_pages = int(recent_pages)
            if self.use_quest:
                bps = -(-self.capacity // block_size)
                nb = batch * bps
                self.page_bounds = [
                    allocate_page_minmax(nb, M.hkv, M.d, device=device)
                    for _ in range(M.n_layers)
                ]
            else:
                self.page_bounds = [None] * M.n_layers

        def sink(self, l, k, v):
            if not (self.use_quest and l >= self.dense_prefix_layers):
                return super().sink(l, k, v)
            T = k.shape[0]
            pos = torch.arange(T, device=k.device)
            slots = self.block_table[0, pos // self.bs].long() * self.bs + pos % self.bs
            kc, vc, ks, vs = self.pool[l]
            kv_ops.reshape_and_cache(k, v, kc, vc, slots, ks, vs)
            k_min, k_max = self.page_bounds[l]
            chunk = 8192
            for s in range(0, T, chunk):
                update_page_minmax(k[s:s + chunk], slots[s:s + chunk], k_min, k_max, self.bs)

        def replicate_prefix(self):
            super().replicate_prefix()
            if self.use_quest:
                n = -(-self.ctx // self.bs)
                src = self.block_table[0, :n].long()
                for l, bounds in enumerate(self.page_bounds):
                    if bounds is None or l < self.dense_prefix_layers:
                        continue
                    for t in bounds:
                        blocks = t[src]
                        for b in range(1, self.B):
                            t.index_copy_(0, self.block_table[b, :n].long(), blocks)

    return ProfileDecodeState()


@contextmanager
def _nvtx_stage(torch, name: str, stage_events: Optional[List[Tuple[str, Any, Any]]] = None):
    torch.cuda.nvtx.range_push(name)
    start = end = None
    if stage_events is not None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    try:
        yield
    finally:
        if stage_events is not None:
            end.record()
            stage_events.append((name, start, end))
        torch.cuda.nvtx.range_pop()


def profile_decode_step(
    st,
    stage_events: Optional[List[Tuple[str, Any, Any]]] = None,
    deps: Optional[Dict[str, Any]] = None,
):
    """Execute one NVTX-annotated decode step across all layers and stages.

    Emits `torch.cuda.nvtx.range_push` / `range_pop` for:
      `Decode_Step` -> `Embedding`, `RMSNorm`, `QKV_Proj`, `RoPE_KV_Append`,
      (`Quest_Page_Select` +) `Paged_Attention`, `O_Proj`, `MLP_SwiGLU`, `LM_Head`.
    """
    if deps is None:
        deps = _import_gpu_deps()
    torch = deps["torch"]
    F = deps["F"]
    rms_norm = deps["rms_norm"]
    kv_ops = deps["kv_ops"]
    apply_rope = deps["apply_rope"]
    paged_sparse_decode = deps["paged_sparse_decode"]
    pages_for_budget = deps["pages_for_budget"]
    quest_page_scores = deps["quest_page_scores"]
    quest_select_pages = deps["quest_select_pages"]

    M, B, pos = st.M, st.B, st.positions
    use_quest = getattr(st, "use_quest", False)
    dense_prefix = getattr(st, "dense_prefix_layers", 2)
    quest_budget = getattr(st, "quest_budget", 0.25)
    recent_pages = getattr(st, "recent_pages", 1)
    page_bounds = getattr(st, "page_bounds", None)

    with _nvtx_stage(torch, OUTER_STAGE, stage_events):
        st.seq_lens.copy_(torch.where(st.active, pos + 1, pos).int())
        blk = st.block_table.gather(1, (pos // st.bs).view(B, 1)).view(B).long()
        slots = torch.where(st.active, blk * st.bs + pos % st.bs,
                            torch.full_like(pos, -1))

        with _nvtx_stage(torch, "Embedding", stage_events):
            h = F.embedding(st.input_ids, M.embed_w)

        top_k_pages = None
        if use_quest:
            top_k_pages = pages_for_budget(quest_budget, st.capacity, st.bs, recent_pages)

        for l in range(M.n_layers):
            L = M.layers[l]
            layer_quest = use_quest and (l >= dense_prefix)

            with _nvtx_stage(torch, "RMSNorm", stage_events):
                x = rms_norm(h, L["ln1"], M.eps)

            with _nvtx_stage(torch, "QKV_Proj", stage_events):
                q = F.linear(x, L["q"], L["qb"]).view(B, M.hq, M.d)
                k = F.linear(x, L["k"], L["kb"]).view(B, M.hkv, M.d)
                v = F.linear(x, L["v"], L["vb"]).view(B, M.hkv, M.d)

            if L["qn"] is not None:
                with _nvtx_stage(torch, "RMSNorm", stage_events):
                    q = rms_norm(q, L["qn"], M.eps)
                    k = rms_norm(k, L["kn"], M.eps)

            with _nvtx_stage(torch, "RoPE_KV_Append", stage_events):
                q = apply_rope(q, M.cos, M.sin, pos)
                k = apply_rope(k, M.cos, M.sin, pos)
                v = v.contiguous()
                kc, vc, ks, vs = st.pool[l]
                k_min, k_max = (
                    page_bounds[l]
                    if (layer_quest and page_bounds is not None)
                    else (None, None)
                )
                kv_ops.reshape_and_cache(
                    k, v, kc, vc, slots, ks, vs, k_min=k_min, k_max=k_max
                )

            if layer_quest:
                with _nvtx_stage(torch, "Quest_Page_Select", stage_events):
                    scores = quest_page_scores(
                        q, k_min, k_max, st.block_table, st.seq_lens, st.bs,
                        recent_pages=recent_pages, max_seq_len=st.capacity,
                    )
                    selected = quest_select_pages(scores, top_k_pages)
                with _nvtx_stage(torch, "Paged_Attention", stage_events):
                    o = paged_sparse_decode(
                        q, kc, vc, st.block_table, st.seq_lens, selected,
                        scale=M.scale, k_scale=ks, v_scale=vs,
                    )
            else:
                with _nvtx_stage(torch, "Paged_Attention", stage_events):
                    o = kv_ops.paged_gqa_decode(
                        q, kc, vc, st.block_table, st.seq_lens, st.capacity,
                        scale=M.scale, k_scale=ks, v_scale=vs,
                    )

            with _nvtx_stage(torch, "O_Proj", stage_events):
                h = h + F.linear(o.reshape(B, -1), L["o"], L["ob"])

            with _nvtx_stage(torch, "RMSNorm", stage_events):
                x = rms_norm(h, L["ln2"], M.eps)

            with _nvtx_stage(torch, "MLP_SwiGLU", stage_events):
                h = h + F.linear(
                    F.silu(F.linear(x, L["gate"])) * F.linear(x, L["up"]),
                    L["down"],
                )

        with _nvtx_stage(torch, "RMSNorm", stage_events):
            x = rms_norm(h, M.norm_w, M.eps)

        with _nvtx_stage(torch, "LM_Head", stage_events):
            logits = F.linear(x, M.lm_head)

        nxt = logits.argmax(-1)
        st.input_ids.copy_(torch.where(st.active, nxt, st.input_ids))
        st.token_log.index_copy_(0, st.step_idx.clamp(max=st.gen), nxt.view(1, B))
        adv = st.active.long()
        st.step_idx.add_(adv.max())
        st.positions.add_(adv)


def _aggregate_cuda_events(stage_events: List[Tuple[str, Any, Any]]) -> List[Dict[str, Any]]:
    totals = {}  # type: Dict[str, float]
    counts = {}  # type: Dict[str, int]
    order = []  # type: List[str]
    for name, start, end in stage_events:
        ms = float(start.elapsed_time(end))
        if name not in totals:
            totals[name] = 0.0
            counts[name] = 0
            order.append(name)
        totals[name] += ms
        counts[name] += 1

    decode_ms = totals.get(OUTER_STAGE, 0.0)
    rows = []  # type: List[Dict[str, Any]]
    for name in order:
        total_ms = totals[name]
        inst = counts[name]
        rows.append({
            "range": name,
            "time_pct": (100.0 * total_ms / decode_ms) if decode_ms > 0 else 0.0,
            "total_ms": total_ms,
            "instances": inst,
            "avg_ms": total_ms / max(inst, 1),
        })
    return rows


def run_worker(args) -> Dict[str, Any]:
    deps = _import_gpu_deps()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise SystemExit("profile_e2e_nsys.py requires a CUDA GPU")
    if args.method.endswith("fp8") and args.block_size % 64 != 0:
        raise SystemExit("--method {} requires --block-size divisible by 64".format(args.method))

    device = "cuda"
    max_len = args.ctx + args.warmup + args.profile_steps
    hf, overrides = deps["load_model"](args.model, max_len, device=device)
    M = deps["Model"](hf, max_len)
    gpu_name = torch.cuda.get_device_name(0)
    print(
        "{}: {} layers, Hq {}, Hkv {}, D {}, gpu {}, method {}, B={}, ctx={}{}".format(
            args.model, M.n_layers, M.hq, M.hkv, M.d, gpu_name, args.method,
            args.batch, args.ctx,
            ", rope override {}".format(overrides) if overrides else "",
        ),
        flush=True,
    )

    g = torch.Generator().manual_seed(args.seed + args.ctx)
    prompt = torch.randint(0, M.vocab, (args.ctx,), generator=g).to(device)
    pre = deps["PrefillCache"](M, prompt)

    gen = args.warmup + args.profile_steps
    st = _make_profile_state(
        deps, M, args.method, args.batch, args.ctx, gen, args.block_size, device,
        args.quest_budget, args.dense_prefix_layers, args.recent_pages, args.seed,
    )
    pre.fill(st, device)
    st.replicate_prefix()
    st.start(pre.logits)

    stage_events = [] if args.cuda_events else None

    with torch.no_grad():
        for _ in range(args.warmup):
            profile_decode_step(st, stage_events=None, deps=deps)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.profile_steps):
            profile_decode_step(st, stage_events=stage_events, deps=deps)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    if args.cuda_events:
        rows = _aggregate_cuda_events(stage_events)
        md = format_profile_markdown(
            [{
                "method": args.method,
                "batch": args.batch,
                "ctx": args.ctx,
                "profile_steps": args.profile_steps,
                "stages": rows,
            }],
            metadata={
                "model": args.model,
                "method": args.method,
                "gpu": gpu_name,
                "profile_steps": args.profile_steps,
                "warmup": args.warmup,
                "block_size": args.block_size,
                "quest_budget": args.quest_budget,
                "dense_prefix_layers": args.dense_prefix_layers,
                "recent_pages": args.recent_pages,
                "source": "cuda_events",
            },
        )
        print("\n" + md)
        return {"gpu": gpu_name, "stages": rows}

    print("profiled {} steps (after {} warmup steps)".format(args.profile_steps, args.warmup))
    return {"gpu": gpu_name, "stages": []}


# --------------------------------------------------------------------------- nsys sweep driver

def run_nsys_sweep(args):
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]

    if args.cuda_events:
        return _run_cuda_events_sweep(args, batches, contexts)

    nsys_path = shutil.which(args.nsys_bin)
    if not nsys_path:
        raise SystemExit(
            "nsys binary {!r} not found in PATH; install Nsight Systems or pass "
            "--cuda-events for lightweight CUDA-event breakdown".format(args.nsys_bin)
        )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    rep_dir = os.path.abspath(args.rep_dir)
    os.makedirs(rep_dir, exist_ok=True)

    records = []  # type: List[Dict[str, Any]]
    gpu_name = None
    for ctx in contexts:
        for batch in batches:
            rep_prefix = os.path.join(
                rep_dir,
                "nsys_{}_b{}_ctx{}".format(args.method, batch, ctx),
            )
            rep_path = rep_prefix + ".nsys-rep"
            worker_cmd = [
                sys.executable, os.path.abspath(__file__),
                "--model", args.model,
                "--method", args.method,
                "--batch", str(batch),
                "--ctx", str(ctx),
                "--profile-steps", str(args.profile_steps),
                "--warmup", str(args.warmup),
                "--block-size", str(args.block_size),
                "--quest-budget", str(args.quest_budget),
                "--dense-prefix-layers", str(args.dense_prefix_layers),
                "--recent-pages", str(args.recent_pages),
                "--seed", str(args.seed),
            ]
            nsys_cmd = [
                nsys_path, "profile",
                "--trace=cuda,nvtx,osrt",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--force-overwrite=true",
                "-o", rep_prefix,
            ] + worker_cmd

            print("== nsys profile: method={} batch={} ctx={} ==".format(
                args.method, batch, ctx), flush=True)
            try:
                prof_res = subprocess.run(
                    nsys_cmd, check=True, capture_output=True, text=True
                )
                if gpu_name is None:
                    m = re.search(r"gpu ([^,\n]+)", prof_res.stdout)
                    if m:
                        gpu_name = m.group(1).strip()
                stats_cmd = [
                    nsys_path, "stats",
                    "--report", "nvtx_sum",
                    "--format", "csv",
                    rep_path,
                ]
                stats_res = subprocess.run(
                    stats_cmd, check=True, capture_output=True, text=True
                )
                stages = parse_nvtx_csv(stats_res.stdout)
                if not stages:
                    raise RuntimeError("empty nvtx_sum CSV output")
                records.append({
                    "method": args.method,
                    "batch": batch,
                    "ctx": ctx,
                    "profile_steps": args.profile_steps,
                    "rep_path": rep_path,
                    "stages": stages,
                })
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                err_msg = getattr(exc, "stderr", "") or str(exc)
                err_line = err_msg.strip().splitlines()[-1] if err_msg.strip() else str(exc)
                print("WARNING: failed batch={} ctx={}: {}".format(batch, ctx, err_line),
                      file=sys.stderr, flush=True)
                records.append({
                    "method": args.method,
                    "batch": batch,
                    "ctx": ctx,
                    "profile_steps": args.profile_steps,
                    "error": err_line,
                })

    metadata = {
        "model": args.model,
        "method": args.method,
        "gpu": gpu_name,
        "profile_steps": args.profile_steps,
        "warmup": args.warmup,
        "block_size": args.block_size,
        "quest_budget": args.quest_budget,
        "dense_prefix_layers": args.dense_prefix_layers,
        "recent_pages": args.recent_pages,
        "source": "nsys stats --report nvtx_sum",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    md = format_profile_markdown(records, metadata)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print("\nwrote profile report -> {}".format(args.out))


def _run_cuda_events_sweep(args, batches: List[int], contexts: List[int]):
    deps = _import_gpu_deps()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise SystemExit("profile_e2e_nsys.py requires a CUDA GPU")
    if args.method.endswith("fp8") and args.block_size % 64 != 0:
        raise SystemExit("--method {} requires --block-size divisible by 64".format(args.method))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = "cuda"
    max_len = max(contexts) + args.warmup + args.profile_steps
    hf, _ = deps["load_model"](args.model, max_len, device=device)
    M = deps["Model"](hf, max_len)
    gpu_name = torch.cuda.get_device_name(0)

    records = []  # type: List[Dict[str, Any]]
    for ctx in contexts:
        g = torch.Generator().manual_seed(args.seed + ctx)
        prompt = torch.randint(0, M.vocab, (ctx,), generator=g).to(device)
        pre = deps["PrefillCache"](M, prompt)
        for batch in batches:
            print("== cuda-events: method={} batch={} ctx={} ==".format(
                args.method, batch, ctx), flush=True)
            try:
                gen = args.warmup + args.profile_steps
                st = _make_profile_state(
                    deps, M, args.method, batch, ctx, gen, args.block_size, device,
                    args.quest_budget, args.dense_prefix_layers, args.recent_pages, args.seed,
                )
                pre.fill(st, device)
                st.replicate_prefix()
                st.start(pre.logits)
                stage_events = []  # type: List[Tuple[str, Any, Any]]
                with torch.no_grad():
                    for _ in range(args.warmup):
                        profile_decode_step(st, stage_events=None, deps=deps)
                    torch.cuda.synchronize()
                    for _ in range(args.profile_steps):
                        profile_decode_step(st, stage_events=stage_events, deps=deps)
                    torch.cuda.synchronize()
                records.append({
                    "method": args.method,
                    "batch": batch,
                    "ctx": ctx,
                    "profile_steps": args.profile_steps,
                    "stages": _aggregate_cuda_events(stage_events),
                })
                del st
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as exc:
                print("WARNING: OOM at batch={} ctx={}: {}".format(batch, ctx, exc),
                      file=sys.stderr, flush=True)
                torch.cuda.empty_cache()
                records.append({
                    "method": args.method,
                    "batch": batch,
                    "ctx": ctx,
                    "profile_steps": args.profile_steps,
                    "error": "OOM",
                })
        del pre
        torch.cuda.empty_cache()

    metadata = {
        "model": args.model,
        "method": args.method,
        "gpu": gpu_name,
        "profile_steps": args.profile_steps,
        "warmup": args.warmup,
        "block_size": args.block_size,
        "quest_budget": args.quest_budget,
        "dense_prefix_layers": args.dense_prefix_layers,
        "recent_pages": args.recent_pages,
        "source": "cuda_events",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    md = format_profile_markdown(records, metadata)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print("\nwrote profile report -> {}".format(args.out))


# --------------------------------------------------------------------------- CLI

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="HF model id (default Qwen/Qwen3-8B) or 'tiny-random'")
    ap.add_argument("--method", choices=METHODS, default="paged_fp16",
                    help="KV cache / attention method to profile")
    ap.add_argument("--batch", type=int, default=1, help="batch size for single-config mode")
    ap.add_argument("--ctx", type=int, default=4096, help="prompt length for single-config mode")
    ap.add_argument("--profile-steps", type=int, default=5,
                    help="steady-state decode steps bracketed by cudaProfilerStart/Stop")
    ap.add_argument("--warmup", type=int, default=3, help="untimed decode steps before profiling")
    ap.add_argument("--block-size", type=int, default=64, help="paged KV block size")
    ap.add_argument("--quest-budget", type=float, default=0.25,
                    help="Quest page budget fraction (for quest_fp16/quest_fp8)")
    ap.add_argument("--dense-prefix-layers", type=int, default=2,
                    help="leading layers kept dense when using Quest")
    ap.add_argument("--recent-pages", type=int, default=1,
                    help="trailing local pages always kept by Quest")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--cuda-events", action="store_true",
                    help="measure per-stage GPU time with CUDA events around NVTX ranges")
    ap.add_argument("--nsys-sweep", action="store_true",
                    help="driver mode: run nsys profile + nsys stats across --batches x --contexts")
    ap.add_argument("--batches", default="1,16,64", help="comma-separated batch sizes for sweep")
    ap.add_argument("--contexts", default="4096,65536",
                    help="comma-separated context lengths for sweep")
    ap.add_argument("--out", default=os.path.join("profiles", "INFERENCE_PROFILE.md"),
                    help="Markdown output path for --nsys-sweep")
    ap.add_argument("--rep-dir", default="profiles",
                    help="directory for .nsys-rep files created by --nsys-sweep")
    ap.add_argument("--nsys-bin", default="nsys", help="path to nsys executable")
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.nsys_sweep:
        run_nsys_sweep(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
