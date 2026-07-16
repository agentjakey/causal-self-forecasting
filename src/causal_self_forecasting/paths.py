"""Filesystem layout for runs and datasets.

One module owns the layout so that a filename is never spelled out in two places. Every
run writes the same set of names, which is what `csf verify run` relies on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .config import repo_root

TRIAL_MANIFEST = "trial_manifest.jsonl"
CANDIDATE_SETS = "candidate_sets.jsonl"
FORECASTS = "forecasts.jsonl"
FORECAST_COMMITMENTS = "forecast_commitments.jsonl"
SELECTION_REVEALS = "selection_reveals.jsonl"
OBSERVATIONS = "observations.parquet"
SCORES = "scores.json"
RUN_MANIFEST = "run_manifest.json"
ENVIRONMENT = "environment.json"
ARTIFACT_HASHES = "artifact_hashes.json"
RUN_LOG = "run.log.jsonl"


def data_dir() -> Path:
    return repo_root() / "data"


def processed_dir() -> Path:
    return data_dir() / "processed"


def manifests_dir() -> Path:
    return data_dir() / "manifests"


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"


def directions_dir() -> Path:
    return artifacts_dir() / "directions"


def adapters_dir() -> Path:
    return artifacts_dir() / "adapters"


def results_dir() -> Path:
    return repo_root() / "results"


def runs_dir() -> Path:
    return results_dir() / "runs"


def public_dir() -> Path:
    return results_dir() / "public"


def run_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def private_dir(run_id: str) -> Path:
    """Directory for payloads and salts that must never be published.

    Kept inside the run directory but under a name the `.gitignore` excludes, so that a
    private payload and its run stay together on disk without either being committed.
    """
    return run_dir(run_id) / "private_payloads"


def new_run_id(prefix: str) -> str:
    """Build a sortable run id such as `smoke-20260715T214800Z`."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in prefix)
    return f"{cleaned}-{stamp}"


def ensure_run_dir(run_id: str) -> Path:
    target = run_dir(run_id)
    target.mkdir(parents=True, exist_ok=True)
    return target
