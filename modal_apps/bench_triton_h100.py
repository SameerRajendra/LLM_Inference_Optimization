"""Run the Triton decode tests and benchmark on a Modal H100 (~$1 per run).

    modal run modal_apps/bench_triton_h100.py

Builds the CUDA extension for sm_90a, runs tests/test_triton_decode.py, then
benchmarks/bench_triton_decode.py. The markdown tables stream to the log and the
CSV is printed at the end: copy it into results/triton/ and commit.
"""
import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

# Same stack as the earlier H100 results (torch 2.8.0 / CUDA 12.8). The -devel image
# carries nvcc for the extension build.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "pytest", "ninja", "pybind11", "setuptools", "wheel", "packaging")
    # Modal's add_python build was made with clang, so setuptools would link with clang++, which this
    # image lacks (and torch warns it is not the compiler it was built with). Build with gcc.
    .env({"CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -pthread -shared"})
    .workdir("/app")
)
# Whitelist, not the whole repo: it also holds personal files that must not leave the machine.
# Only .git/HEAD and .git/refs ride along (enough to stamp the commit; the 60 MB history is not
# needed), and the locally built extension (*.so) stays behind: the container builds its own.
for name in ("csrc", "sparse_kv", "triton_kernels", "tests", "benchmarks", ".git/refs"):
    image = image.add_local_dir(REPO_ROOT / name, f"/app/{name}", ignore=["**/__pycache__", "**/*.so"])
for name in ("setup.py", "pyproject.toml", ".git/HEAD"):
    image = image.add_local_file(REPO_ROOT / name, f"/app/{name}")

app = modal.App("triton-decode-bench")


@app.function(image=image, gpu="H100", timeout=20 * 60)
def bench() -> str:
    def run(*cmd):
        subprocess.run(cmd, cwd="/app", check=True)

    run("nvidia-smi")
    run("python", "setup.py", "build_ext", "--inplace")
    run("python", "-m", "pytest", "tests/test_triton_decode.py", "-q")
    run("python", "benchmarks/bench_triton_decode.py")
    return next(Path("/app/results/triton").glob("*/decode.csv")).read_text()


@app.local_entrypoint()
def main():
    print(bench.remote())
