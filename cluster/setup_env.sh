#!/usr/bin/env bash
# One-time environment setup on the Slurm cluster (no sudo, no docker).
# Run from the repo root on a LOGIN node:
#
#   bash cluster/setup_env.sh
#
# Everything is user-space: Lmod modules + a Python venv + header-only CUTLASS.
# Adjust the knobs below (or override via env vars) to match your cluster.
set -euo pipefail

# ----------------------------- knobs -----------------------------------------
CUDA_MODULE="${CUDA_MODULE:-cuda12.4}"        # `module avail cuda` to list
PYTHON_MODULE="${PYTHON_MODULE:-}"             # e.g. python/3.11 (empty = system python3)
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/llmopt}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
CUTLASS_TAG="${CUTLASS_TAG:-v3.9.2}"
# HF model cache belongs on scratch, not $HOME (quota). $SCRATCH if defined.
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"
# ------------------------------------------------------------------------------

echo "== [1/6] Modules =="
if command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE" || {
        echo "ERROR: could not 'module load $CUDA_MODULE'."
        echo "       Run 'module avail cuda' and re-run with CUDA_MODULE=<name>."
        exit 1
    }
    [ -n "$PYTHON_MODULE" ] && module load "$PYTHON_MODULE"
else
    echo "WARN: no Lmod 'module' command; assuming nvcc is already on PATH."
fi
command -v nvcc >/dev/null || { echo "ERROR: nvcc not on PATH after module load."; exit 1; }
nvcc --version | tail -1

echo "== [2/6] Python venv at $VENV_DIR =="
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -V
pip install --upgrade pip setuptools wheel ninja pybind11

echo "== [3/6] PyTorch ($TORCH_SPEC, $TORCH_INDEX) =="
pip install "$TORCH_SPEC" --extra-index-url "$TORCH_INDEX"
# Triton ships inside the Linux torch wheel (pytorch-triton) — no extra install.
python -c "import triton; print('triton', triton.__version__)"

echo "== [4/6] Project Python deps =="
[ -f requirements.txt ] && pip install -r requirements.txt
pip install "huggingface_hub[cli]"

echo "== [5/6] CUTLASS headers (header-only, $CUTLASS_TAG) =="
if [ ! -d third_party/cutlass ]; then
    git clone --depth 1 --branch "$CUTLASS_TAG" \
        https://github.com/NVIDIA/cutlass.git third_party/cutlass
else
    echo "third_party/cutlass already present — skipping clone."
fi

echo "== [6/6] Build CUDA extension (sm_90a; compiles fine on a login node) =="
mkdir -p cluster/logs
pip install -e . --no-build-isolation -v

echo
echo "== Setup done. Next steps =="
echo "  1. Gated Llama weights:   hf auth login    (or export HF_TOKEN=...)"
echo "  2. Sanity check (login):  python cluster/check_env.py"
echo "  3. Full check (GPU node): sbatch cluster/slurm/test.sbatch"
echo "  HF_HOME=$HF_HOME"
