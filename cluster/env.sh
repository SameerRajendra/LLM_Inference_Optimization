# Sourced by every Slurm job script — one place to fix cluster specifics.
# No venv: this cluster's user-site python3 already carries torch/triton/
# transformers (installed by cluster/setup_env.sh into ~/.local). nvcc is on
# PATH by default here, so CUDA_MODULE is empty unless a node needs it
# (confirmed module name on this cluster: cuda12.4).
CUDA_MODULE="${CUDA_MODULE:-}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"

# PINNED interpreter — see the comment in cluster/setup_env.sh. Job scripts must
# call "$PYTHON_BIN", never bare `python`/`python3`: those resolve differently on
# login vs compute nodes, and the CUDA extension is ABI-locked to one of them.
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.9}"

# We build in place (setup.py build_ext --inplace) rather than `pip install -e .`,
# so nothing drops a .pth that makes `import sparse_kv` work from anywhere.
# Scripts under benchmarks/ and cluster/ get their OWN directory on sys.path[0],
# not the repo root — without this they cannot import the package.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "$CUDA_MODULE" ] && command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE"
fi
