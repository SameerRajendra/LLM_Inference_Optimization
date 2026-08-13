#!/bin/bash
#SBATCH --job-name=llmopt-build
#SBATCH --partition=gpu            # ADJUST: `sinfo` to list partitions
#SBATCH --gres=gpu:1               # ADJUST if H100s need e.g. gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
#
# Rebuild the CUDA extension after a `git pull`. Usually a login-node
#   python setup.py build_ext --inplace
# is enough (sm_90a compiles without a GPU); use this when login nodes
# forbid heavy compilation.
#   sbatch cluster/slurm/build.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

python setup.py build_ext --inplace -v
python cluster/check_env.py --require-gpu
