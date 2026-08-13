#!/usr/bin/env bash
# One-time environment setup on the Slurm cluster (no sudo, no docker, no venv —
# targets the shared user-site Python 3.9 already on this cluster, which
# already ships torch/triton/transformers).
#
#   bash cluster/setup_env.sh
#
# ------------------------------ knobs -----------------------------------------
CUDA_MODULE="${CUDA_MODULE:-}"                 # confirmed module name: cuda12.4 — leave
                                                # empty if nvcc is already on PATH
                                                # (Bright Cluster Manager profile does
                                                # this by default on this cluster)
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUTLASS_TAG="${CUTLASS_TAG:-v3.9.2}"
# HF model cache belongs on scratch, not $HOME (quota).
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"
# --------------------------------------------------------------------------------
set -euo pipefail

echo "== [1/5] CUDA toolkit =="
if [ -n "$CUDA_MODULE" ] && command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE" || echo "WARN: 'module load $CUDA_MODULE' failed — checking PATH anyway."
fi
command -v nvcc >/dev/null || {
    echo "ERROR: nvcc not on PATH. Find the module with 'module avail cuda' and"
    echo "       re-run: CUDA_MODULE=<name> bash cluster/setup_env.sh"
    exit 1
}
nvcc --version | tail -1

echo "== [2/5] Python build deps (user site) =="
"$PYTHON_BIN" -m pip install --user --upgrade setuptools wheel ninja pybind11

echo "== [3/5] PyTorch =="
if "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
    "$PYTHON_BIN" -c "import torch; print('found torch', torch.__version__, 'cuda', torch.version.cuda)"
else
    echo "torch not found — installing (adjust TORCH_SPEC/TORCH_INDEX if this cluster needs a different build)."
    "$PYTHON_BIN" -m pip install --user "${TORCH_SPEC:-torch}" \
        --extra-index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
fi
# Triton ships inside the Linux torch wheel (pytorch-triton) — no extra install.
"$PYTHON_BIN" -c "import triton; print('triton', triton.__version__)"

echo "== [4/5] Project Python deps + CUTLASS headers ($CUTLASS_TAG) =="
[ -f requirements.txt ] && "$PYTHON_BIN" -m pip install --user -r requirements.txt
"$PYTHON_BIN" -m pip install --user "huggingface_hub[cli]"
if [ ! -d third_party/cutlass ]; then
    git clone --depth 1 --branch "$CUTLASS_TAG" \
        https://github.com/NVIDIA/cutlass.git third_party/cutlass
else
    echo "third_party/cutlass already present — skipping clone."
fi

echo "== [5/5] Build CUDA extension (sm_90a, in place) =="
mkdir -p cluster/logs
"$PYTHON_BIN" setup.py build_ext --inplace -v

echo
echo "== Setup done. Next steps =="
echo "  1. Gated Llama weights:   hf auth login    (or export HF_TOKEN=...)"
echo "  2. Sanity check (login):  python cluster/check_env.py"
echo "  3. Full check (GPU node): sbatch cluster/slurm/test.sbatch"
echo "  HF_HOME=$HF_HOME"
