# Sourced by every Slurm job script — one place to fix cluster specifics.
CUDA_MODULE="${CUDA_MODULE:-cuda12.4}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/llmopt}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"

if command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
