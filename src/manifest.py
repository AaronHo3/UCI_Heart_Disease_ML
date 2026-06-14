"""Lightweight experiment tracking: a reproducibility manifest.

Rather than a heavyweight tracking server, this records the provenance every
result in ``reports/`` needs to be trustworthy: the exact git commit, the random
seed, the library versions, and a content hash of every output artifact. Run
``make manifest`` after regenerating results so reviewers can confirm that the
committed figures and CSVs came from a known, reproducible state.

Output: ``reports/run_manifest.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from pipeline_utils import RANDOM_SEED

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TRACKED_PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "shap"]

# Each analysis maps to the command that produces it (for traceability).
ANALYSES = {
    "phase1_loso": "make loso",
    "phase2_nested_cv": "make nested",
    "phase3_calibration": "make calibrate",
    "phase3_decision_curve": "make dca",
    "phase4_interpretability": "make interpret",
    "phase5_missingness": "make missingness",
    "phase6_conformal": "make conformal",
    "phase7_fairness": "make fairness",
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        return False


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _package_versions() -> dict[str, str]:
    out = {}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _artifact_digest(directory: str, exts: tuple[str, ...]) -> dict[str, str]:
    digest = {}
    if not os.path.isdir(directory):
        return digest
    for name in sorted(os.listdir(directory)):
        if name.endswith(exts):
            path = os.path.join(directory, name)
            digest[path] = _sha256(path)
    return digest


def build_manifest() -> dict:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_sha(),
        "git_dirty": _git_dirty(),
        "random_seed": RANDOM_SEED,
        "package_versions": _package_versions(),
        "analyses": ANALYSES,
        "artifacts": {
            "csv": _artifact_digest(REPORTS_DIR, (".csv",)),
            "figures": _artifact_digest(FIGURES_DIR, (".png",)),
        },
    }


def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    manifest = build_manifest()
    path = os.path.join(REPORTS_DIR, "run_manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"git commit : {manifest['git_commit'][:12]} (dirty={manifest['git_dirty']})")
    print(f"seed       : {manifest['random_seed']}")
    print(f"artifacts  : {len(manifest['artifacts']['csv'])} csv, "
          f"{len(manifest['artifacts']['figures'])} figures")
    print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
