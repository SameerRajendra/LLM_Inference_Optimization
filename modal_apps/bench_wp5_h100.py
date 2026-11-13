"""Run WP-5 H100 benchmarks on Modal (TP=2, TP=4 NVLink decode + Nsight Systems).

Runs the Modal-compatible parts of WP-5 (~$2-4 total on H100! instead of renting
a 4x H100 VM):
  1. TP=2 decode overlap + CUDA Graph over NCCL (`gpu="H100!:2"`)
  2. TP=4 decode overlap + CUDA Graph over NCCL (`gpu="H100!:4"`)
  3. E2E Nsight Systems (`nsys`) NVTX + `cuda_gpu_kern_sum` profile (`gpu="H100!"`)

Note: `ncu` PMU hardware counters (`profiles/NCU_PROFILE.md`) are blocked in
Modal containers (`ERR_NVGPUCTRPERM`) and still require a 1x H100 VM/bare-metal
node (~10 min).

Usage:
    modal run modal_apps/bench_wp5_h100.py
    modal run modal_apps/bench_wp5_h100.py --no-tp4 --no-nsys   # TP=2 only (~$0.65)
    modal run modal_apps/bench_wp5_h100.py --no-tp2 --no-tp4    # nsys only (~$0.65)
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "command -v nsys >/dev/null 2>&1 || ("
        "apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
        "cuda-nsight-systems-12-8 || "
        "apt-get install -y --no-install-recommends nsight-systems-cli || true"
        ") && rm -rf /var/lib/apt/lists/*"
    )
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy",
        "pytest",
        "ninja",
        "pybind11",
        "setuptools",
        "wheel",
        "packaging",
        "transformers==4.57.6",
        "accelerate",
    )
    .env({
        "CC": "gcc",
        "CXX": "g++",
        "LDSHARED": "g++ -pthread -shared",
        "HF_HOME": "/cache/hf",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .workdir("/app")
)

for name in ("csrc", "sparse_kv", "triton_kernels", "tests", "benchmarks", "docs", ".git/refs"):
    image = image.add_local_dir(
        REPO_ROOT / name, f"/app/{name}", ignore=["**/__pycache__", "**/*.so"]
    )
for name in ("setup.py", "pyproject.toml", "README.md", "RESULTS.md", ".git/HEAD", ".git/packed-refs"):
    p = REPO_ROOT / name
    if p.exists():
        image = image.add_local_file(p, f"/app/{name}")

app = modal.App("wp5-h100-bench")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("huggingface")] if os.environ.get("HF_SECRET") else []


def _build_ext():
    subprocess.run(["nvidia-smi"], cwd="/app", check=True)
    subprocess.run(["python", "setup.py", "build_ext", "--inplace"], cwd="/app", check=True)


def _collect_new_results(start_ts: float) -> dict:
    out = {}
    results_dir = Path("/app/results")
    if not results_dir.exists():
        return out
    for ext in ("*.json", "*.csv", "*.md", "*.txt"):
        for path in results_dir.rglob(ext):
            if path.stat().st_mtime >= start_ts:
                out[str(path.relative_to("/app"))] = path.read_text()
    return out


def _run_tp_suite(tp_size: int, shape: str, kv_mode: str, batches: str, contexts: str,
                  steps: int, warmup: int, with_nsys_trace: bool) -> dict:
    start = time.time()
    _build_ext()
    subprocess.run(["nvidia-smi", "topo", "-m"], cwd="/app", check=False)

    subprocess.run(
        [
            "python",
            "benchmarks/bench_tp_decode.py",
            "--tp", str(tp_size),
            "--shape", shape,
            "--kv-mode", kv_mode,
            "--batches", batches,
            "--contexts", contexts,
            "--steps", str(steps),
            "--warmup", str(warmup),
        ],
        cwd="/app",
        check=True,
    )

    nsys_bin = shutil.which("nsys")
    if with_nsys_trace and nsys_bin:
        rep_prefix = f"/tmp/tp{tp_size}_decode"
        rep_path = rep_prefix + ".nsys-rep"
        nsys_cmd = [
            nsys_bin,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--trace-fork-before-exec=true",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            "-o", rep_prefix,
            "torchrun",
            "--standalone",
            f"--nproc_per_node={tp_size}",
            "benchmarks/bench_tp_decode.py",
            "--shape", shape,
            "--kv-mode", "paged_fp8",
            "--batches", "16",
            "--contexts", "4096",
            "--steps", "16",
            "--warmup", "4",
            "--nsys-bracket",
            "--no-save",
        ]
        prof = subprocess.run(nsys_cmd, cwd="/app", check=False)
        if prof.returncode == 0 and Path(rep_path).exists():
            stats_dir = Path("/app/results/profiles")
            stats_dir.mkdir(parents=True, exist_ok=True)
            chunks = []
            for report in ("nvtx_sum", "cuda_gpu_kern_sum"):
                res = subprocess.run(
                    [nsys_bin, "stats", "--report", report, rep_path],
                    cwd="/app",
                    capture_output=True,
                    text=True,
                    check=False,
                )
                chunks.append(f"=== nsys stats --report {report} (TP={tp_size}) ===\n{res.stdout}\n{res.stderr}")
            (stats_dir / f"h100_tp{tp_size}_nsys_stats.txt").write_text("\n".join(chunks))

    return _collect_new_results(start)


@app.function(image=image, gpu="H100!:2", timeout=30 * 60, memory=32 * 1024)
def bench_tp2(shape: str = "qwen3-8b", kv_mode: str = "both", batches: str = "1,4,16,64",
              contexts: str = "4096,16384", steps: int = 64, warmup: int = 8,
              with_nsys_trace: bool = True) -> dict:
    return _run_tp_suite(2, shape, kv_mode, batches, contexts, steps, warmup, with_nsys_trace)


@app.function(image=image, gpu="H100!:4", timeout=30 * 60, memory=32 * 1024)
def bench_tp4(shape: str = "qwen3-8b", kv_mode: str = "both", batches: str = "1,4,16,64",
              contexts: str = "4096,16384", steps: int = 64, warmup: int = 8,
              with_nsys_trace: bool = False) -> dict:
    return _run_tp_suite(4, shape, kv_mode, batches, contexts, steps, warmup, with_nsys_trace)


@app.function(image=image, gpu="H100!", timeout=30 * 60, volumes={"/cache": hf_cache},
              secrets=SECRETS, memory=48 * 1024)
def bench_nsys(model: str = "Qwen/Qwen3-8B", method: str = "paged_fp8",
               batches: str = "1,16,64", contexts: str = "4096,65536",
               profile_steps: int = 5, warmup: int = 3, block_size: int = 64) -> dict:
    start = time.time()
    _build_ext()

    nsys_bin = shutil.which("nsys")
    if not nsys_bin:
        raise RuntimeError("nsys was not found in the container")
    subprocess.run([nsys_bin, "--version"], cwd="/app", check=True)

    sys.path.insert(0, "/app")
    import torch
    from benchmarks.profile_e2e_nsys import format_profile_markdown, parse_nvtx_csv

    gpu_name = torch.cuda.get_device_name(0)
    batch_list = [int(x) for x in batches.split(",") if x.strip()]
    ctx_list = [int(x) for x in contexts.split(",") if x.strip()]

    out_dir = Path("/app/results/profiles")
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path("/tmp/nsys_reps")
    rep_dir.mkdir(parents=True, exist_ok=True)

    records = []
    kern_summaries = []
    for ctx in ctx_list:
        for batch in batch_list:
            rep_prefix = str(rep_dir / f"nsys_{method}_b{batch}_ctx{ctx}")
            rep_path = rep_prefix + ".nsys-rep"
            cmd = [
                nsys_bin,
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--force-overwrite=true",
                "-o", rep_prefix,
                "python",
                "benchmarks/profile_e2e_nsys.py",
                "--model", model,
                "--method", method,
                "--batch", str(batch),
                "--ctx", str(ctx),
                "--profile-steps", str(profile_steps),
                "--warmup", str(warmup),
                "--block-size", str(block_size),
            ]
            print(f"== nsys: {method} B={batch} ctx={ctx} ==", flush=True)
            prof = subprocess.run(cmd, cwd="/app", capture_output=True, text=True, check=False)
            if prof.returncode != 0 or not Path(rep_path).exists():
                combined = (prof.stderr or "") + "\n" + (prof.stdout or "")
                if "OutOfMemoryError" in combined or "CUDA out of memory" in combined:
                    err_line = "OOM"
                else:
                    tail = combined.strip().splitlines()
                    err_line = tail[-1] if tail else f"exit {prof.returncode}"
                print(f"WARNING: B={batch} ctx={ctx} failed: {err_line}", flush=True)
                records.append({
                    "method": method,
                    "batch": batch,
                    "ctx": ctx,
                    "profile_steps": profile_steps,
                    "error": err_line,
                })
                continue

            nvtx_csv = subprocess.run(
                [nsys_bin, "stats", "--report", "nvtx_sum", "--format", "csv", rep_path],
                cwd="/app",
                capture_output=True,
                text=True,
                check=True,
            )
            stages = parse_nvtx_csv(nvtx_csv.stdout)
            records.append({
                "method": method,
                "batch": batch,
                "ctx": ctx,
                "profile_steps": profile_steps,
                "rep_path": rep_path,
                "stages": stages,
            })

            kern = subprocess.run(
                [nsys_bin, "stats", "--report", "cuda_gpu_kern_sum", rep_path],
                cwd="/app",
                capture_output=True,
                text=True,
                check=False,
            )
            kern_summaries.append(
                f"=== cuda_gpu_kern_sum: {method} B={batch} ctx={ctx} ===\n{kern.stdout}\n{kern.stderr}"
            )

    hf_cache.commit()

    metadata = {
        "model": model,
        "method": method,
        "gpu": gpu_name,
        "profile_steps": profile_steps,
        "warmup": warmup,
        "block_size": block_size,
        "source": "nsys stats --report nvtx_sum (--sample=none --cpuctxsw=none)",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "h100_nsys_profile.md").write_text(format_profile_markdown(records, metadata))
    if kern_summaries:
        (out_dir / "h100_nsys_kern_sum.txt").write_text("\n".join(kern_summaries))

    return _collect_new_results(start)


def _write_back(files: dict) -> None:
    for rel, text in files.items():
        dst = REPO_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        print("wrote", dst)


@app.local_entrypoint()
def main(
    no_tp2: bool = False,
    no_tp4: bool = False,
    no_nsys: bool = False,
    shape: str = "qwen3-8b",
    kv_mode: str = "both",
    tp_batches: str = "1,4,16,64",
    tp_contexts: str = "4096,16384",
    model: str = "Qwen/Qwen3-8B",
    method: str = "paged_fp8",
    nsys_batches: str = "1,16,64",
    nsys_contexts: str = "4096,65536",
):
    if not no_tp2:
        _write_back(bench_tp2.remote(shape, kv_mode, tp_batches, tp_contexts))
    if not no_tp4:
        _write_back(bench_tp4.remote(shape, kv_mode, tp_batches, tp_contexts))
    if not no_nsys:
        _write_back(bench_nsys.remote(model, method, nsys_batches, nsys_contexts))
