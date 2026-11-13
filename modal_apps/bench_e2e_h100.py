"""Full H100 verification & benchmark sweep (WP-1 through WP-5 + SASS dump) in one container.

    modal run modal_apps/bench_e2e_h100.py
    modal run modal_apps/bench_e2e_h100.py --model meta-llama/Llama-3.1-8B   # needs HF_SECRET=1

Smoke run first (one case per engine, a few minutes): checks that the image builds and that the
flashinfer-python pip resolves next to vllm==0.10.2 still has the plan()/run() API used by
bench_vs_engines.py, before paying for the full sweep:

    modal run modal_apps/bench_e2e_h100.py --batches 1 --vs-batches 1 --contexts 4096 --steps 4 --no-save --no-decode --no-quest --no-profile

Steps (all in one container, ~1-2 h of H100 for the full sweep):
  1. build the CUDA extension for sm_90a
  2. pytest suite (cudagraph_compile, gqa_tc, paged_gqa, quest_sparse, triton_decode, wp5_scripts)
  3. SASS instruction-mix summary via benchmarks/profile_ncu.py --sass
  4. benchmarks/bench_triton_decode.py (sdpa, cuda_v4/fp8, cuda_tc/fp8, triton fp16/fp8)
  5. benchmarks/bench_e2e_serving.py  (SDPA eager / compile vs sparse_kv paged fp16 / fp8)
  6. benchmarks/profile_e2e_nsys.py --nsys-sweep --cuda-events (per-stage % breakdown)
  7. benchmarks/bench_vs_engines.py --engine vllm,flashinfer, with step 5's paged rows merged in
     via --sparse-kv-results (sparse_kv is not measured twice)
  8. benchmarks/quality_gate_quest.py --parts kernel,quality (64K kernel speedup + Qwen3-8B gate)
The result JSON / CSV / MD files written under results/ are copied back into the local results/.

Model weights are cached in the Modal Volume "hf-cache" (downloaded once). Gated models
(meta-llama/*) need a Modal secret named "huggingface" holding HF_TOKEN; export HF_SECRET=1 when
running so it is attached.
"""
import os
import subprocess
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

# vLLM pins its torch build (0.10.2 -> torch 2.8.0 / CUDA 12.8, the stack of the earlier H100
# results), so install it first and build our extension against that torch. The -devel image
# carries nvcc (and cuobjdump) for the extension build and SASS summary.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm==0.10.2")
    .pip_install("flashinfer-python")
    .pip_install("numpy", "pytest", "ninja", "pybind11", "setuptools", "wheel", "packaging",
                 "transformers==4.57.6", "accelerate")
    # Modal's add_python build was made with clang; build the extension with gcc.
    .env({"CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -pthread -shared",
          "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .workdir("/app")
)
# Whitelist, not the whole repo: it also holds personal files that must not leave the machine.
for name in ("csrc", "sparse_kv", "triton_kernels", "tests", "benchmarks", "docs", ".git/refs"):
    image = image.add_local_dir(REPO_ROOT / name, f"/app/{name}", ignore=["**/__pycache__", "**/*.so"])
for name in ("setup.py", "pyproject.toml", "README.md", "RESULTS.md", ".git/HEAD", ".git/packed-refs"):
    p = REPO_ROOT / name
    if p.exists():
        image = image.add_local_file(p, f"/app/{name}")

app = modal.App("e2e-decode-bench")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("huggingface")] if os.environ.get("HF_SECRET") else []


# memory: bench_e2e_serving keeps one ctx's prefill K/V in host RAM (~10 GB at 64K, Qwen3-8B).
@app.function(image=image, gpu="H100", timeout=3 * 60 * 60, volumes={"/cache": hf_cache},
              secrets=SECRETS, memory=48 * 1024)
def bench(model: str, batches: str, contexts: str, vs_batches: str, steps: int,
          no_save: bool = False, with_decode: bool = True, with_quest: bool = True,
          quest_parts: str = "kernel,quality", with_profile: bool = True) -> dict:
    start = time.time()

    def run(*cmd, check: bool = True):
        return subprocess.run(cmd, cwd="/app", check=check)

    def new_results(bench_name):
        found = [p for p in Path("/app/results", bench_name).glob("*.json")
                 if p.stat().st_mtime >= start]
        return sorted(found, key=lambda p: p.stat().st_mtime)

    run("nvidia-smi")
    run("python", "setup.py", "build_ext", "--inplace")
    run("python", "-m", "pytest",
        "tests/test_cudagraph_compile.py", "tests/test_gqa_tc.py",
        "tests/test_paged_gqa.py", "tests/test_quest_sparse.py",
        "tests/test_triton_decode.py", "tests/test_wp5_scripts.py", "-v")

    sass_cmd = ["python", "benchmarks/profile_ncu.py", "--sass"]
    if not no_save:
        Path("/app/results/profiles").mkdir(parents=True, exist_ok=True)
        sass_cmd += ["--sass-out", "results/profiles/h100_sass.md"]
    run(*sass_cmd, check=False)

    if with_decode and not no_save:
        run("python", "benchmarks/bench_triton_decode.py", "--batches", "1,4")

    common = ["--model", model, "--contexts", contexts, "--steps", str(steps)]
    save_flag = ["--no-save"] if no_save else []
    run("python", "benchmarks/bench_e2e_serving.py", "--batches", batches, *common, *save_flag)
    hf_cache.commit()  # keep the downloaded weights even if a later step fails

    if with_profile and not no_save:
        run("python", "benchmarks/profile_e2e_nsys.py", "--nsys-sweep", "--cuda-events",
            "--model", model, "--method", "paged_fp8", "--batches", "1,16,64",
            "--contexts", "4096,65536",
            "--out", "results/profiles/h100_inference_profile.md", check=False)

    e2e = new_results("e2e_serving")
    merge = ["--sparse-kv-results", str(e2e[-1])] if e2e else []
    run("python", "benchmarks/bench_vs_engines.py", "--engine", "vllm,flashinfer",
        "--batches", vs_batches, *common, *merge, *save_flag)

    if with_quest:
        res = run("python", "benchmarks/quality_gate_quest.py",
                  "--parts", quest_parts, "--model", model, *save_flag, check=False)
        if res.returncode != 0:
            print(f"warning: quality_gate_quest exited {res.returncode} (saved JSONs preserved)")

    out = {}
    if not no_save:
        for ext in ("*.json", "*.csv", "*.md"):
            for path in Path("/app/results").rglob(ext):
                if path.stat().st_mtime >= start:
                    out[str(path.relative_to("/app"))] = path.read_text()
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-8B", batches: str = "1,4,8,16,32,64",
         contexts: str = "4096,16384,65536", vs_batches: str = "1,4,8,16,32", steps: int = 64,
         no_save: bool = False, no_decode: bool = False, no_quest: bool = False,
         quest_parts: str = "kernel,quality", no_profile: bool = False):
    files = bench.remote(model, batches, contexts, vs_batches, steps, no_save,
                         with_decode=not no_decode, with_quest=not no_quest,
                         quest_parts=quest_parts, with_profile=not no_profile)
    for rel, text in files.items():
        dst = REPO_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        print("wrote", dst)
