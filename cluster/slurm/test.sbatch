#!/bin/bash
#SBATCH --job-name=llmopt-test
#SBATCH --partition=gpu            # ADJUST
#SBATCH --gres=gpu:1               # ADJUST
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
#
# Correctness gate on one H100: env check + full pytest suite.
#   sbatch cluster/slurm/test.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

python cluster/check_env.py --require-gpu
pytest tests/ -v
