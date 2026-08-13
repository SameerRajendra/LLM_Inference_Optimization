#!/usr/bin/env python
"""Environment sanity check for the decode-engine workstreams (WS1-WS4).

Run on a login node (GPU checks are skipped with a warning) or on a compute
node with ``--require-gpu`` (GPU checks become hard failures):

    python cluster/check_env.py                # login node
    python cluster/check_env.py --require-gpu  # inside a Slurm GPU job

Exit code 0 = ready; 1 = at least one hard failure.

NOTE: keep this file Python-3.9 compatible (no ``X | None`` annotations).
"""
import argparse
import os
import shutil
import subprocess
import sys

FAILURES = []


def report(ok, label, detail="", hard=True):
    mark = "PASS" if ok else ("FAIL" if hard else "WARN")
    print("[{}] {} {}".format(mark, label, "- " + detail if detail else ""))
    if not ok and hard:
        FAILURES.append(label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true",
                        help="treat missing GPU / wrong arch as hard failures")
    args = parser.parse_args()

    print("== Python ==")
    report(sys.version_info >= (3, 9), "python >= 3.9", sys.version.split()[0])

    print("== CUDA toolkit ==")
    nvcc = shutil.which("nvcc")
    if nvcc:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
        line = out.stdout.strip().splitlines()[-1] if out.stdout else "?"
        report(True, "nvcc on PATH", line)
    else:
        report(False, "nvcc on PATH", "module load cuda/12.x first")

    print("== PyTorch ==")
    try:
        import torch
    except ImportError as exc:
        report(False, "import torch", str(exc))
        return finish()
    report(True, "torch", torch.__version__)
    report(torch.version.cuda is not None, "torch CUDA build",
           "cu" + str(torch.version.cuda))
    report(hasattr(torch, "float8_e4m3fn"), "FP8 dtype (float8_e4m3fn)",
           "needed for WS3")

    print("== GPU ==")
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        report(True, "GPU visible", "{} sm_{}{}".format(name, cap[0], cap[1]))
        report(cap == (9, 0), "Hopper (sm_90)",
               "kernels are compiled -arch=sm_90a", hard=args.require_gpu)
        report(torch.cuda.is_bf16_supported(), "bf16 supported")
        free, total = torch.cuda.mem_get_info()
        report(True, "HBM", "{:.0f} / {:.0f} GiB free".format(
            free / 2**30, total / 2**30))
    else:
        report(False, "GPU visible",
               "no CUDA device (fine on a login node)", hard=args.require_gpu)

    print("== Triton (WS2) ==")
    try:
        import triton
        report(True, "triton", triton.__version__)
    except ImportError as exc:
        report(False, "triton", str(exc))

    print("== CUDA extension (WS1/WS3/WS4) ==")
    try:
        from sparse_kv import _C
        syms = [s for s in dir(_C) if not s.startswith("_")]
        report(True, "sparse_kv._C importable", ", ".join(syms))
    except ImportError as exc:
        report(False, "sparse_kv._C importable",
               "python {}.{} - python setup.py build_ext --inplace ({})".format(
                   sys.version_info.major, sys.version_info.minor, exc))

    print("== CUTLASS headers (WS1/WS3) ==")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    header = os.path.join(root, "third_party", "cutlass",
                          "include", "cutlass", "cutlass.h")
    report(os.path.isfile(header), "third_party/cutlass",
           "cluster/setup_env.sh clones it", hard=False)

    print("== HF stack (benchmarks) ==")
    try:
        import transformers
        report(True, "transformers", transformers.__version__)
    except ImportError as exc:
        report(False, "transformers", str(exc), hard=False)
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface (default)")
    report(True, "HF_HOME", hf_home)
    report(bool(os.environ.get("HF_TOKEN")), "HF_TOKEN set",
           "needed only for gated meta-llama weights", hard=False)

    return finish()


def finish():
    print()
    if FAILURES:
        print("NOT READY - failures: " + "; ".join(FAILURES))
        return 1
    print("READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
