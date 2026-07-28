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
# Observations are written as JSONL, not parquet. Every other record in a run is JSONL, the
# volume is small, and a hashable line-oriented file needs none of the pandas machinery a
# parquet writer would pull in. The parquet name is kept above only as the documented long
# term target; nothing writes it yet.
OBSERVATION_RECORDS = "observations.jsonl"
RESOLUTION_FAILURES = "resolution_failures.jsonl"
RESOLUTION_MANIFEST = "resolution.json"
SCORES = "scores.json"
SCORE_RECORDS = "score_records.jsonl"
RUN_MANIFEST = "run_manifest.json"
ENVIRONMENT = "environment.json"
ARTIFACT_HASHES = "artifact_hashes.json"
RUN_LOG = "run.log.jsonl"

# Systems-benchmark artifacts. Named distinctly from the trial artifacts above so that a
# benchmark run directory can never be mistaken for an experiment run directory.
BENCHMARK = "benchmark.json"
BENCHMARK_ITEMS = "benchmark_items.jsonl"

# BlueDot state-dependence artifacts. Prefixed so a study run directory is never confused with
# a benchmark trial run: the two hold different targets, and `csf trials resolve` must not read
# a study run's observations as if they carried `delta_margin`.
STATE_AUDIT_RUN_MANIFEST = "state_audit_run.json"
STATE_AUDIT_OBSERVATIONS = "state_audit_observations.jsonl"
STATE_AUDIT_FAILURES = "state_audit_failures.jsonl"
STATE_AUDIT_CANDIDATE_SETS = "state_audit_candidate_sets.jsonl"
STATE_AUDIT_STATE_REFS = "state_audit_state_refs.jsonl"
STATE_AUDIT_STATES = "state_audit_states.npz"
STATE_AUDIT_CLEAN_PASS = "state_audit_clean_pass.jsonl"


def data_dir() -> Path:
    return repo_root() / "data"


def processed_dir() -> Path:
    return data_dir() / "processed"


def manifests_dir() -> Path:
    return data_dir() / "manifests"


def prompt_manifests_dir() -> Path:
    """Where frozen prompt-role manifests live.

    Kept beside the task manifests and outside the git-ignored `data/` subdirectories, because
    a prompt manifest is a frozen split. Committing it is what freezes it, and a split that
    only exists on one machine is not a preregistered split.
    """
    return data_dir() / "prompt_manifests"


def prompt_manifest_path(manifest_id: str) -> Path:
    return prompt_manifests_dir() / f"{manifest_id}.json"


def direction_manifests_dir() -> Path:
    """Where direction-family manifests live.

    Tracked, unlike the `.npz` vectors under `artifacts/directions/`. The manifest is small and
    is the thing that makes a family citable and regenerable; the vectors are reproducible from
    it plus the pinned model, so they stay out of version control.
    """
    return data_dir() / "direction_manifests"


def direction_manifest_path(family_id: str) -> Path:
    return direction_manifests_dir() / f"{family_id}.json"


def calibration_plans_dir() -> Path:
    """Where frozen calibration plans live.

    Tracked, like the prompt and direction manifests. A plan says what would count as a passing
    ratio, and it has to be written down before the numbers exist or it is not a plan. The state
    norms, observations, and decisions that a real calibration produces belong in ignored run
    directories instead.
    """
    return data_dir() / "calibration_plans"


def calibration_plan_path(plan_id: str) -> Path:
    return calibration_plans_dir() / f"{plan_id}.json"


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
