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


def _git_dir():
    d = os.path.join(REPO_ROOT, ".git")
    if os.path.isfile(d):                     # worktree: .git is a pointer file
        try:
            with open(d) as f:
                line = f.read().strip()
            if line.startswith("gitdir:"):
                d = os.path.normpath(
                    os.path.join(REPO_ROOT, line.split(":", 1)[1].strip()))
        except OSError:
            return ""
    return d if os.path.isdir(d) else ""


def _head_from_files():
    """Read HEAD without the git binary.

    Compute nodes on this cluster have no `git`, and a run with no commit
    attached is not evidence. Parsing .git directly keeps every result pinned.
    Returns (short_sha, branch).
    """
    gd = _git_dir()
    if not gd:
        return "", ""
    try:
        with open(os.path.join(gd, "HEAD")) as f:
            head = f.read().strip()
    except OSError:
        return "", ""
    if not head.startswith("ref:"):
        return head[:7], "detached"
    ref = head.split(":", 1)[1].strip()
    branch = ref.rsplit("/", 1)[-1]
    loose = os.path.join(gd, *ref.split("/"))
    if os.path.isfile(loose):
        try:
            with open(loose) as f:
                return f.read().strip()[:7], branch
        except OSError:
            pass
    packed = os.path.join(gd, "packed-refs")     # ref may only be packed
    if os.path.isfile(packed):
        try:
            with open(packed) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0][:7], branch
        except OSError:
            pass
    return "", branch


def provenance():
    """Everything needed to reproduce or challenge a number."""
    has_git = bool(_git("rev-parse", "--git-dir"))
    if has_git:
        sha = _git("rev-parse", "--short", "HEAD")
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        subject = _git("log", "-1", "--pretty=%s")
        dirty = bool(_git("status", "--porcelain"))
    else:
        # No git binary (GPU compute nodes here). Read .git directly; subject
        # and dirty need object decoding / a full diff, so they stay unknown
        # rather than being silently reported as clean.
        sha, branch = _head_from_files()
        subject = ""
        dirty = None

    info = {
        "utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": sha,
        "git_branch": branch,
        "git_subject": subject,
        "git_dirty": dirty,          # None = could not be determined
        "git_available": has_git,
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
    if prov["git_dirty"] is True:
        payload["WARNING"] = ("working tree was dirty - this number does not "
                              "correspond to a clean commit")
    elif prov["git_dirty"] is None:
        payload["NOTE"] = ("no git binary on this node - SHA read from .git, "
                           "but working-tree cleanliness was not verified")

    out_dir = os.path.join(REPO_ROOT, "results", bench)
    os.makedirs(out_dir, exist_ok=True)
    fname = "{}__{}__{}.json".format(
        prov["utc"].replace(":", "").replace("-", ""), safe,
        prov["git_sha"] or "nogit")
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    if not quiet:
        print("\nsaved -> results/{}/{}  [{}]".format(
            bench, fname, prov["git_sha"] or "NO SHA"))
        if prov["git_dirty"] is True:
            print("WARNING: working tree dirty - commit before trusting this run")
    return path
