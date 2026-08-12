# Cluster runbook (Slurm, H100 SXM, no sudo / no docker)

Development loop: **author locally → commit → push to GitHub → `git pull` on the
cluster → build/test/benchmark via Slurm.** Nothing is ever executed locally.

## One-time setup (login node)

```bash
git clone https://github.com/SameerRajendra/LLM_Inference_Optimization.git
cd LLM_Inference_Optimization
git checkout decode-engine

bash cluster/setup_env.sh          # modules + venv + torch cu124 + CUTLASS + build
hf auth login                      # only for gated meta-llama weights
python cluster/check_env.py        # login-node sanity (GPU checks skipped)
```

If a module name differs on your cluster, override the knobs:

```bash
CUDA_MODULE=cuda/12.4 PYTHON_MODULE=python/3.11 bash cluster/setup_env.sh
```

All cluster-specific settings live in two places only: the knobs at the top of
`cluster/setup_env.sh` (one-time) and `cluster/env.sh` (sourced by every job).
Edit the `#SBATCH --partition/--gres` lines in `cluster/slurm/*.sbatch` once to
match `sinfo` output.

## Every iteration

```bash
git pull                                   # get the code I authored
sbatch cluster/slurm/build.sbatch          # rebuild extension (or login-node pip install -e .)
sbatch cluster/slurm/test.sbatch           # correctness gate: check_env + pytest on H100
sbatch cluster/slurm/bench.sbatch          # isolated-kernel bandwidth benchmark
RUN_E2E=1 sbatch --export=ALL cluster/slurm/bench.sbatch   # + end-to-end model decode
```

Logs land in `cluster/logs/<job>-<id>.out` (git-ignored). After a run, commit
the new `results/*.csv|json` and push — that is how measured numbers get back
into the repo and the README.

Interactive debugging session on a GPU node:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
source cluster/env.sh
python cluster/check_env.py --require-gpu
```

## Profiling without sudo

- **Nsight Systems** (`nsys profile python ...`) works unprivileged.
- **Nsight Compute** (`ncu`) needs GPU performance-counter access. Without
  sudo you cannot flip the driver's `NVreg_RestrictProfilingToAdminUsers`
  setting — but most HPC clusters already run with counters enabled or expose
  a `module load nsight-compute`. If `ncu` reports `ERR_NVGPUCTRPERM`, ask the
  cluster admins to enable profiling on the GPU partition; meanwhile the
  benchmarks' effective-bandwidth accounting (bytes-moved / measured-latency)
  is the fallback metric.
- Copy `.nsys-rep` / `.ncu-rep` files off the cluster to inspect in the local
  GUI; they are git-ignored (too large) — use `scp`, not git.

## Model weights

Primary benchmark target: `meta-llama/Llama-3.1-8B` (gated — needs `HF_TOKEN`).
Secondary: `Qwen/Qwen3-8B` (ungated, same 32Q/8KV/128-dim GQA geometry).
Weights cache under `HF_HOME` (defaults to `$SCRATCH/hf_cache`), never in the
repo or `$HOME`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvcc: command not found` | `module avail cuda`, then `CUDA_MODULE=<name> bash cluster/setup_env.sh` |
| Build OK but `sparse_kv._C` import fails on GPU node | toolkit/driver mismatch — load the same `CUDA_MODULE` in the job (already done by `cluster/env.sh`) |
| `no kernel image is available` at runtime | GPU is not sm_90 — request the H100 partition/constraint in the sbatch file |
| `ERR_NVGPUCTRPERM` from `ncu` | see "Profiling without sudo" above |
| Gated-repo 403 for Llama | `hf auth login`, accept the license on the model page, or use `Qwen/Qwen3-8B` |
| Slurm job exits instantly, empty log | `cluster/logs/` missing on that clone — `mkdir -p cluster/logs` |
