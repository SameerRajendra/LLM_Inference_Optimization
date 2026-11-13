# CLAUDE.md — Project Instructions (Keep Context Minimal)

## Repo Purpose
Long-context LLM decode engine in CUDA + Triton targeting H100 (`sm_90a`).

## Hard Constraints
- **Python 3.9 compatibility** in `sparse_kv/`, `triton_kernels/`, `benchmarks/`, `tests/`: use `from __future__ import annotations` if typing unions are needed; never use Python 3.10+ runtime syntax (`X | Y` without future annotations, `match/case`).
- **Never run `pip install -e .`** on Slurm clusters (causes recursive pip bug). Build CUDA extension in place:
  `python setup.py build_ext --inplace`
- **CPU test command** (works without a GPU):
  `TRITON_INTERPRET=1 pytest tests/test_triton_decode.py -v`
- **Page contract:** `KV_PAGE_TOKENS` in `csrc/kernels/fp8_common.cuh` and `triton_kernels/gqa_decode.py` must stay identical (currently `64`).
- **No fabricated benchmarks:** Never hand-edit numbers in `RESULTS.md` or `results/`. All metrics must come from runnable scripts via `benchmarks/bench_common.py` or `modal_apps/`.

## Token-Saving Workflow
1. **Only read the specific Work Package (`WP-*`) in `docs/WORK_PACKAGES.md` assigned by the user** plus the exact files listed in that Work Package.
2. **Do NOT read** `results/`, `profiles/`, `LLM_Inference_optimization.pdf`, or `benchmarks/llama_integration_benchmark.py` unless explicitly requested.
3. Run `TRITON_INTERPRET=1 pytest` (or CPU unit tests) before finishing any task.
4. Keep edits minimal and focused; report exact files changed and test commands run.
