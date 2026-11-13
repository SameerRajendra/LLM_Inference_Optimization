"""Unified Modal H100 SXM runner for kernels, E2E serving, multi-GPU TP, and nsys.

Usage:
    modal run modal_apps/h100.py --suite kernel   # build + pytest + SASS + decode microbench (~$1)
    modal run modal_apps/h100.py --suite e2e      # full WP-1..WP-4 + vLLM / FlashInfer + Quest
    modal run modal_apps/h100.py --suite tp2      # WP-5 TP=2 NCCL decode + CUDA Graph (~$0.65)
    modal run modal_apps/h100.py --suite tp4      # WP-5 TP=4 NCCL decode + CUDA Graph (~$1.30)
    modal run modal_apps/h100.py --suite nsys     # WP-5 Nsight Systems NVTX + cuda_gpu_kern_sum (~$0.65)
    modal run modal_apps/h100.py --suite wp5      # tp2 + tp4 + nsys sequentially (~$2.60)
    modal run modal_apps/h100.py --suite all      # e2e + wp5

Smoke check for `--suite e2e`:
    modal run modal_apps/h100.py --suite e2e --batches 1 --vs-batches 1 --contexts 4096 \\
        --steps 4 --no-save --no-decode --no-quest --no-profile

Note: `ncu` hardware PMU counters (`profiles/NCU_PROFILE.md`) are blocked inside Modal
containers (`ERR_NVGPUCTRPERM`) and require a bare-metal or VM H100 node.
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

COMMON_ENV = {
    "CC": "gcc",
    "CXX": "g++",
    "LDSHARED": "g++ -pthread -shared",
    "HF_HOME": "/cache/hf",
    "TOKENIZERS_PARALLELISM": "false",
}


def _mount_repo(img: modal.Image) -> modal.Image:
    for name in ("csrc", "sparse_kv", "triton_kernels", "tests", "benchmarks", "docs", ".git/refs"):
        img = img.add_local_dir(
            REPO_ROOT / name, f"/app/{name}", ignore=["**/__pycache__", "**/*.so"]
        )
    for name in ("setup.py", "pyproject.toml", "README.md", "RESULTS.md", ".git/HEAD", ".git/packed-refs"):
        p = REPO_ROOT / name
        if p.exists():
            img = img.add_local_file(p, f"/app/{name}")
    return img


# Lightweight image for kernels, TP=2/4 NCCL, and Nsight Systems (no vLLM/FlashInfer overhead).
core_image = _mount_repo(
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
    .env(COMMON_ENV)
    .workdir("/app")
)

# Full serving comparison image pinning vLLM 0.10.2 (torch 2.8.0/cu128) + FlashInfer.
engines_image = _mount_repo(
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm==0.10.2")
    .pip_install("flashinfer-python")
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
    .env(COMMON_ENV)
    .workdir("/app")
)

app = modal.App("h100-decode-engine")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("huggingface")] if os.environ.get("HF_SECRET") else []

PYTEST_SUITE = [
    "tests/test_cudagraph_compile.py",
    "tests/test_gqa_tc.py",
    "tests/test_paged_gqa.py",
    "tests/test_quest_sparse.py",
    "tests/test_triton_decode.py",
    "tests/test_wp5_scripts.py",
]


def _run(*cmd: str, check: bool = True):
    return subprocess.run(cmd, cwd="/app", check=check)


def _build_ext() -> None:
    _run("nvidia-smi")
    _run("python", "setup.py", "build_ext", "--inplace")


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


def _new_result_jsons(bench_name: str, start_ts: float) -> list:
    found = [
        p for p in Path("/app/results", bench_name).glob("*.json")
        if p.stat().st_mtime >= start_ts
    ]
    return sorted(found, key=lambda p: p.stat().st_mtime)


@app.function(image=core_image, gpu="H100!", timeout=30 * 60)
def run_kernel(no_save: bool = False) -> dict:
    start = time.time()
    _build_ext()
    _run("python", "-m", "pytest", *PYTEST_SUITE, "-v")

    sass_cmd = ["python", "benchmarks/profile_ncu.py", "--sass"]
    if not no_save:
        Path("/app/results/profiles").mkdir(parents=True, exist_ok=True)
        sass_cmd += ["--sass-out", "results/profiles/h100_sass.md"]
    _run(*sass_cmd, check=False)

    if not no_save:
        _run("python", "benchmarks/bench_triton_decode.py", "--batches", "1,4")
    return {} if no_save else _collect_new_results(start)


@app.function(
    image=engines_image,
    gpu="H100!",
    timeout=3 * 60 * 60,
    volumes={"/cache": hf_cache},
    secrets=SECRETS,
    memory=48 * 1024,
)
def run_e2e(
    model: str = "Qwen/Qwen3-8B",
    batches: str = "1,4,8,16,32,64",
    contexts: str = "4096,16384,65536",
    vs_batches: str = "1,4,8,16,32",
    steps: int = 64,
    no_save: bool = False,
    with_decode: bool = True,
    with_quest: bool = True,
    quest_parts: str = "kernel,quality",
    with_profile: bool = True,
) -> dict:
    start = time.time()
    _build_ext()
    _run("python", "-m", "pytest", *PYTEST_SUITE, "-v")

    sass_cmd = ["python", "benchmarks/profile_ncu.py", "--sass"]
    if not no_save:
        Path("/app/results/profiles").mkdir(parents=True, exist_ok=True)
        sass_cmd += ["--sass-out", "results/profiles/h100_sass.md"]
    _run(*sass_cmd, check=False)

    if with_decode and not no_save:
        _run("python", "benchmarks/bench_triton_decode.py", "--batches", "1,4")

    common = ["--model", model, "--contexts", contexts, "--steps", str(steps)]
    save_flag = ["--no-save"] if no_save else []
    _run("python", "benchmarks/bench_e2e_serving.py", "--batches", batches, *common, *save_flag)
    hf_cache.commit()

    if with_profile and not no_save:
        _run(
            "python",
            "benchmarks/profile_e2e_nsys.py",
            "--nsys-sweep",
            "--cuda-events",
            "--model", model,
            "--method", "paged_fp8",
            "--batches", "1,16,64",
            "--contexts", "4096,65536",
            "--out", "results/profiles/h100_inference_profile.md",
            check=False,
        )

    e2e = _new_result_jsons("e2e_serving", start)
    merge = ["--sparse-kv-results", str(e2e[-1])] if e2e else []
    _run(
        "python",
        "benchmarks/bench_vs_engines.py",
        "--engine", "vllm,flashinfer",
        "--batches", vs_batches,
        *common,
        *merge,
        *save_flag,
    )

    if with_quest:
        res = _run(
            "python",
            "benchmarks/quality_gate_quest.py",
            "--parts", quest_parts,
            "--model", model,
            *save_flag,
            check=False,
        )
        if res.returncode != 0:
            print(f"warning: quality_gate_quest exited {res.returncode} (saved JSONs preserved)")

    return {} if no_save else _collect_new_results(start)


def _run_tp_suite(
    tp_size: int,
    shape: str,
    kv_mode: str,
    batches: str,
    contexts: str,
    steps: int,
    warmup: int,
    with_nsys_trace: bool,
    no_save: bool,
) -> dict:
    start = time.time()
    _build_ext()
    _run("nvidia-smi", "topo", "-m", check=False)

    cmd = [
        "python",
        "benchmarks/bench_tp_decode.py",
        "--tp", str(tp_size),
        "--shape", shape,
        "--kv-mode", kv_mode,
        "--batches", batches,
        "--contexts", contexts,
        "--steps", str(steps),
        "--warmup", str(warmup),
    ]
    if no_save:
        cmd.append("--no-save")
    _run(*cmd)

    nsys_bin = shutil.which("nsys")
    if with_nsys_trace and nsys_bin and not no_save:
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
                chunks.append(
                    f"=== nsys stats --report {report} (TP={tp_size}) ===\n{res.stdout}\n{res.stderr}"
                )
            (stats_dir / f"h100_tp{tp_size}_nsys_stats.txt").write_text("\n".join(chunks))

    return {} if no_save else _collect_new_results(start)


@app.function(image=core_image, gpu="H100!:2", timeout=30 * 60, memory=32 * 1024)
def run_tp2(
    shape: str = "qwen3-8b",
    kv_mode: str = "both",
    batches: str = "1,4,16,64",
    contexts: str = "4096,16384",
    steps: int = 64,
    warmup: int = 8,
    with_nsys_trace: bool = True,
    no_save: bool = False,
) -> dict:
    return _run_tp_suite(2, shape, kv_mode, batches, contexts, steps, warmup, with_nsys_trace, no_save)


@app.function(image=core_image, gpu="H100!:4", timeout=30 * 60, memory=32 * 1024)
def run_tp4(
    shape: str = "qwen3-8b",
    kv_mode: str = "both",
    batches: str = "1,4,16,64",
    contexts: str = "4096,16384",
    steps: int = 64,
    warmup: int = 8,
    with_nsys_trace: bool = False,
    no_save: bool = False,
) -> dict:
    return _run_tp_suite(4, shape, kv_mode, batches, contexts, steps, warmup, with_nsys_trace, no_save)


@app.function(
    image=core_image,
    gpu="H100!",
    timeout=30 * 60,
    volumes={"/cache": hf_cache},
    secrets=SECRETS,
    memory=48 * 1024,
)
def run_nsys(
    model: str = "Qwen/Qwen3-8B",
    method: str = "paged_fp8",
    batches: str = "1,16,64",
    contexts: str = "4096,65536",
    profile_steps: int = 5,
    warmup: int = 3,
    block_size: int = 64,
    no_save: bool = False,
) -> dict:
    start = time.time()
    _build_ext()

    nsys_bin = shutil.which("nsys")
    if not nsys_bin:
        raise RuntimeError("nsys was not found in the container")
    _run(nsys_bin, "--version")

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
            records.append({
                "method": method,
                "batch": batch,
                "ctx": ctx,
                "profile_steps": profile_steps,
                "rep_path": rep_path,
                "stages": parse_nvtx_csv(nvtx_csv.stdout),
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

    if not no_save:
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
    return {} if no_save else _collect_new_results(start)


def _write_back(files: dict) -> None:
    for rel, text in files.items():
        dst = REPO_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        print("wrote", dst)


@app.local_entrypoint()
def main(
    suite: str = "wp5",
    model: str = "Qwen/Qwen3-8B",
    batches: str = "1,4,8,16,32,64",
    contexts: str = "4096,16384,65536",
    vs_batches: str = "1,4,8,16,32",
    steps: int = 64,
    no_save: bool = False,
    no_decode: bool = False,
    no_quest: bool = False,
    quest_parts: str = "kernel,quality",
    no_profile: bool = False,
    shape: str = "qwen3-8b",
    kv_mode: str = "both",
    tp_batches: str = "1,4,16,64",
    tp_contexts: str = "4096,16384",
    method: str = "paged_fp8",
    nsys_batches: str = "1,16,64",
    nsys_contexts: str = "4096,65536",
):
    valid = {"kernel", "e2e", "tp2", "tp4", "nsys", "wp5", "all"}
    if suite not in valid:
        raise SystemExit(f"Unknown --suite {suite!r}; choose from {sorted(valid)}")

    if suite == "kernel":
        _write_back(run_kernel.remote(no_save=no_save))
        return

    if suite in ("e2e", "all"):
        _write_back(
            run_e2e.remote(
                model=model,
                batches=batches,
                contexts=contexts,
                vs_batches=vs_batches,
                steps=steps,
                no_save=no_save,
                with_decode=not no_decode,
                with_quest=not no_quest,
                quest_parts=quest_parts,
                with_profile=not no_profile,
            )
        )

    if suite in ("tp2", "wp5", "all"):
        _write_back(
            run_tp2.remote(
                shape=shape,
                kv_mode=kv_mode,
                batches=tp_batches,
                contexts=tp_contexts,
                steps=steps,
                no_save=no_save,
            )
        )

    if suite in ("tp4", "wp5", "all"):
        _write_back(
            run_tp4.remote(
                shape=shape,
                kv_mode=kv_mode,
                batches=tp_batches,
                contexts=tp_contexts,
                steps=steps,
                no_save=no_save,
            )
        )

    if suite in ("nsys", "wp5", "all"):
        _write_back(
            run_nsys.remote(
                model=model,
                method=method,
                batches=nsys_batches,
                contexts=nsys_contexts,
                no_save=no_save,
            )
        )
