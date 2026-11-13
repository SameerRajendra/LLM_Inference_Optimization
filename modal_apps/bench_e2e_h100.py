"""WP-2 on a Modal H100: CUDA-graph tests, real e2e decode benchmark, FlashInfer / vLLM head-to-head.

    modal run modal_apps/bench_e2e_h100.py
    modal run modal_apps/bench_e2e_h100.py --model meta-llama/Llama-3.1-8B   # needs HF_SECRET=1

Smoke run first (one case per engine, a few minutes): checks that the image builds and that the
flashinfer-python pip resolves next to vllm==0.10.2 still has the plan()/run() API used by
bench_vs_engines.py, before paying for the full sweep:

    modal run modal_apps/bench_e2e_h100.py --batches 1 --vs-batches 1 --contexts 4096 --steps 4 --no-save

Steps (all in one container, ~1-2 h of H100 for the full sweep):
  1. build the CUDA extension for sm_90a
  2. pytest tests/test_cudagraph_compile.py (capture/replay, fullgraph compile, block recycling)
  3. benchmarks/bench_e2e_serving.py  (SDPA eager / compile vs sparse_kv paged fp16 / fp8)
  4. benchmarks/bench_vs_engines.py --engine vllm,flashinfer, with step 3's paged rows merged in
     via --sparse-kv-results (sparse_kv is not measured twice)
The result JSONs written by bench_common.save_result are copied back into the local results/.

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
# carries nvcc for the extension build.
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
for name in ("csrc", "sparse_kv", "triton_kernels", "tests", "benchmarks", ".git/refs"):
    image = image.add_local_dir(REPO_ROOT / name, f"/app/{name}", ignore=["**/__pycache__", "**/*.so"])
for name in ("setup.py", "pyproject.toml", ".git/HEAD"):
    image = image.add_local_file(REPO_ROOT / name, f"/app/{name}")

app = modal.App("e2e-decode-bench")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("huggingface")] if os.environ.get("HF_SECRET") else []


# memory: bench_e2e_serving keeps one ctx's prefill K/V in host RAM (~10 GB at 64K, Qwen3-8B).
@app.function(image=image, gpu="H100", timeout=3 * 60 * 60, volumes={"/cache": hf_cache},
              secrets=SECRETS, memory=48 * 1024)
def bench(model: str, batches: str, contexts: str, vs_batches: str, steps: int,
          no_save: bool = False) -> dict:
    start = time.time()

    def run(*cmd):
        subprocess.run(cmd, cwd="/app", check=True)

    def new_results(bench_name):
        found = [p for p in Path("/app/results", bench_name).glob("*.json")
                 if p.stat().st_mtime >= start]
        return sorted(found, key=lambda p: p.stat().st_mtime)

    run("nvidia-smi")
    run("python", "setup.py", "build_ext", "--inplace")
    run("python", "-m", "pytest", "tests/test_cudagraph_compile.py", "tests/test_gqa_tc.py", "-v")
    common = ["--model", model, "--contexts", contexts, "--steps", str(steps)]
    save_flag = ["--no-save"] if no_save else []
    run("python", "benchmarks/bench_e2e_serving.py", "--batches", batches, *common, *save_flag)
    hf_cache.commit()  # keep the downloaded weights even if a later step fails

    e2e = new_results("e2e_serving")
    merge = ["--sparse-kv-results", str(e2e[-1])] if e2e else []
    run("python", "benchmarks/bench_vs_engines.py", "--engine", "vllm,flashinfer",
        "--batches", vs_batches, *common, *merge, *save_flag)

    out = {}
    if not no_save:
        for path in Path("/app/results").rglob("*.json"):
            if path.stat().st_mtime >= start:
                out[str(path.relative_to("/app"))] = path.read_text()
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-8B", batches: str = "1,4,8,16,32,64",
         contexts: str = "4096,16384,65536", vs_batches: str = "1,4,8,16,32", steps: int = 64,
         no_save: bool = False):
    files = bench.remote(model, batches, contexts, vs_batches, steps, no_save)
    for rel, text in files.items():
        dst = REPO_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        print("wrote", dst)
