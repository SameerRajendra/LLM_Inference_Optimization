#!/bin/bash
#SBATCH --job-name=llmopt-bench
#SBATCH --partition=gpu            # ADJUST
#SBATCH --gres=gpu:1               # ADJUST
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out
#
# Benchmarks on one H100. Isolated-kernel bandwidth run always executes;
# the end-to-end model run only with RUN_E2E=1 (downloads weights to HF_HOME).
#   sbatch cluster/slurm/bench.sbatch
#   RUN_E2E=1 MODEL_ID=meta-llama/Llama-3.1-8B sbatch --export=ALL cluster/slurm/bench.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

export MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B}"

python cluster/check_env.py --require-gpu
python benchmarks/profile_kernel_vs_sdpa.py

if [ "${RUN_E2E:-0}" = "1" ]; then
    python benchmarks/llama_integration_benchmark.py
fi
