"""
Shared result-recording for every benchmark in this repo.

Every number this project reports must be traceable: which commit produced it,
on which GPU, with which torch/CUDA, on what date. A printed table in a
terminal is not evidence; a JSON file pinned to a git SHA is.

Layout written:
    results/<bench>/<UTC timestamp>__<label>__<sha>.json

Use from a benchmark:
    from bench_common import provenance, save_result
    save_result("kernel_vs_sdpa", rows, label="stage1-bank-conflict")

NOTE: keep this file Python-3.9 compatible (no `X | None` annotations).
"""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    try:
        out = subprocess.run(["git"] + list(args), cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def provenance():
    """Everything needed to reproduce or challenge a number."""
    info = {
        "utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_subject": _git("log", "-1", "--pretty=%s"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "host": platform.node(),
        "python": sys.version.split()[0],
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            _, total = torch.cuda.mem_get_info()
            info["gpu"] = torch.cuda.get_device_name(0)
            info["sm"] = "sm_{}{}".format(cap[0], cap[1])
            info["hbm_gb"] = round(total / 1e9, 1)
    except Exception:
        pass
    return info


def save_result(bench, rows, label="", extra=None, quiet=False):
    """Write one benchmark run to results/<bench>/ and return the path.

    `label` names the experiment (e.g. "stage1-bank-conflict"); it defaults to
    the current commit subject so a run is never anonymous.
    """
    prov = provenance()
    if not label:
        label = prov.get("git_subject", "") or "unlabelled"
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in label)[:60]

    payload = {"benchmark": bench, "label": label,
               "provenance": prov, "rows": rows}
    if extra:
        payload["extra"] = extra
    if prov["git_dirty"]:
        payload["WARNING"] = ("working tree was dirty - this number does not "
                              "correspond to a clean commit")

    out_dir = os.path.join(REPO_ROOT, "results", bench)
    os.makedirs(out_dir, exist_ok=True)
    fname = "{}__{}__{}.json".format(
        prov["utc"].replace(":", "").replace("-", ""), safe,
        prov["git_sha"] or "nogit")
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    if not quiet:
        print("\nsaved -> results/{}/{}".format(bench, fname))
        if prov["git_dirty"]:
            print("WARNING: working tree dirty - commit before trusting this run")
    return path
