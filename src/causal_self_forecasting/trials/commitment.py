"""The forecast commitment protocol.

The order of operations is the whole point, so the code enforces it rather than trusting the
caller to follow it:

1. A forecast is validated and written.
2. A salted commitment is written. The salt goes to a private directory that git ignores.
3. Only now may a selection seed exist. `commit_forecast` refuses to run if one already
   does, because a seed that predates the commitment means the forecaster could have known
   which candidate would be chosen.
4. The seed selects a candidate deterministically.
5. The intervention is applied and observed.
6. The salt is revealed and the commitment is recomputed and checked.

This does not defend against an experimenter with full control of the machine, who could
simply rerun everything. It does make casual after-the-fact editing detectable, and it makes
the sequence reproducible and publicly checkable, which is what it is for. That boundary is
stated in `docs/claim_boundaries.md` rather than being implied away here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..hashing import (
    append_jsonl,
    atomic_write_text,
    commitment_hash,
    new_salt,
    read_jsonl,
    salt_from_hex,
    salt_to_hex,
    sha256_hex,
    verify_commitment,
)
from ..logging_utils import info
from ..paths import (
    CANDIDATE_SETS,
    FORECAST_COMMITMENTS,
    FORECASTS,
    SELECTION_REVEALS,
    private_dir,
    run_dir,
)
from ..reproducibility import derive_seed
from ..schemas import CandidateSet, ForecastCommitment, ForecastRecord, SelectionReveal

SELECTION_SEED_FILENAME = "selection_seed.txt"


class ProtocolOrderError(RuntimeError):
    """Raised when the commit/select/reveal steps are attempted out of order."""


class CommitmentVerificationError(RuntimeError):
    """Raised when a revealed commitment does not recompute."""


def _salt_path(run_id: str, trial_id: str, method_id: str) -> Path:
    return private_dir(run_id) / "salts" / f"{trial_id}.{method_id}.salt"


def selection_seed_path(run_id: str) -> Path:
    return private_dir(run_id) / SELECTION_SEED_FILENAME


def forecast_payload(forecast: ForecastRecord) -> dict[str, Any]:
    """The exact object that gets hashed.

    Defined in one place so the committer and the verifier can never disagree about what was
    committed to.
    """
    return forecast.model_dump(mode="json")


def selection_seed_exists(run_id: str) -> bool:
    return selection_seed_path(run_id).exists()


def commit_forecast(
    run_id: str,
    forecast: ForecastRecord,
    allow_existing_seed: bool = False,
) -> ForecastCommitment:
    """Write a forecast and its salted commitment.

    Refuses to run once a selection seed exists for the run. `allow_existing_seed` exists
    only for tests that need to construct an out-of-order state deliberately, and it is
    never set by the CLI.
    """
    if not allow_existing_seed and selection_seed_exists(run_id):
        raise ProtocolOrderError(
            f"run {run_id} already has a selection seed at {selection_seed_path(run_id)}; "
            "forecasts committed after the seed exists are not blinded and will not be scored"
        )

    salt = new_salt()
    digest = commitment_hash(forecast_payload(forecast), salt)

    directory = run_dir(run_id)
    append_jsonl(directory / FORECASTS, forecast)

    commitment = ForecastCommitment(
        trial_id=forecast.trial_id,
        method_id=forecast.method_id,
        commitment_hash=digest,
        forecast_ref=FORECASTS,
    )
    append_jsonl(directory / FORECAST_COMMITMENTS, commitment)

    salt_file = _salt_path(run_id, forecast.trial_id, forecast.method_id)
    salt_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(salt_file, salt_to_hex(salt))

    info(
        "committed forecast",
        run_id=run_id,
        trial_id=forecast.trial_id,
        method_id=forecast.method_id,
        commitment=digest[:23],
    )
    return commitment


def write_selection_seed(run_id: str, seed_hex: str, force: bool = False) -> Path:
    """Record the selection seed for a run.

    Rejected if no commitments exist yet: a seed written before any forecast is committed
    would let a forecaster see which candidate is coming.
    """
    commitments_file = run_dir(run_id) / FORECAST_COMMITMENTS
    if not commitments_file.exists():
        raise ProtocolOrderError(
            f"run {run_id} has no commitments yet; a selection seed must not exist before "
            "forecasts are committed"
        )
    target = selection_seed_path(run_id)
    if target.exists() and not force:
        raise ProtocolOrderError(
            f"run {run_id} already has a selection seed; rewriting it would allow reselecting "
            "until a convenient candidate came up. Pass force only to repair a broken run."
        )
    return atomic_write_text(target, seed_hex.strip())


def read_selection_seed(run_id: str, seed_file: str | Path | None = None) -> str:
    path = Path(seed_file) if seed_file is not None else selection_seed_path(run_id)
    if not path.exists():
        raise ProtocolOrderError(f"no selection seed found at {path}")
    seed_hex = path.read_text(encoding="utf-8").strip()
    if not seed_hex:
        raise ProtocolOrderError(f"selection seed at {path} is empty")
    return seed_hex


def select_candidate_index(seed_hex: str, candidate_set: CandidateSet) -> int:
    """Choose one candidate deterministically from the seed and the trial id.

    Deriving per trial from the seed, rather than consuming a stream, means selection for any
    single trial can be recomputed on its own without replaying the run in order.
    """
    count = len(candidate_set.candidates)
    return derive_seed("selection", seed_hex, candidate_set.trial_id) % count


def reveal_selection(
    run_id: str,
    candidate_set: CandidateSet,
    commitment: ForecastCommitment,
    forecast: ForecastRecord,
    seed_hex: str,
) -> SelectionReveal:
    """Select, reveal the salt, and verify the commitment.

    A failed verification is recorded as `verified=False` rather than raised. The record is
    the evidence; erasing it by crashing would be the wrong response to the one situation
    this protocol exists to detect. Downstream, the exporter refuses to publish a run with
    any unverified reveal.
    """
    index = select_candidate_index(seed_hex, candidate_set)
    selected = candidate_set.candidates[index]

    salt_file = _salt_path(run_id, forecast.trial_id, forecast.method_id)
    if not salt_file.exists():
        raise ProtocolOrderError(
            f"no salt found for trial {forecast.trial_id} method {forecast.method_id} at "
            f"{salt_file}; the forecast cannot be revealed"
        )
    salt = salt_from_hex(salt_file.read_text(encoding="utf-8"))
    verified = verify_commitment(forecast_payload(forecast), salt, commitment.commitment_hash)

    reveal = SelectionReveal(
        trial_id=forecast.trial_id,
        method_id=forecast.method_id,
        selection_seed_hash=sha256_hex(seed_hex.encode("utf-8")),
        selected_intervention_id=selected.intervention_id,
        selected_index=index,
        salt_hex=salt_to_hex(salt),
        commitment_hash=commitment.commitment_hash,
        verified=verified,
    )
    append_jsonl(run_dir(run_id) / SELECTION_REVEALS, reveal)
    return reveal


def read_commitments(run_id: str) -> list[ForecastCommitment]:
    path = run_dir(run_id) / FORECAST_COMMITMENTS
    if not path.exists():
        return []
    return [ForecastCommitment.model_validate(row) for row in read_jsonl(path)]


def read_forecasts(run_id: str) -> list[ForecastRecord]:
    path = run_dir(run_id) / FORECASTS
    if not path.exists():
        return []
    return [ForecastRecord.model_validate(row) for row in read_jsonl(path)]


def read_reveals(run_id: str) -> list[SelectionReveal]:
    path = run_dir(run_id) / SELECTION_REVEALS
    if not path.exists():
        return []
    return [SelectionReveal.model_validate(row) for row in read_jsonl(path)]


def read_candidate_sets(run_id: str) -> list[CandidateSet]:
    path = run_dir(run_id) / CANDIDATE_SETS
    if not path.exists():
        return []
    return [CandidateSet.model_validate(row) for row in read_jsonl(path)]


def verify_run_commitments(run_id: str) -> dict[str, Any]:
    """Recompute every commitment in a run from its revealed salt.

    This is the check a third party runs. It recomputes the hash from the forecast on disk
    and the revealed salt, so an edited forecast fails even though its commitment record
    still looks well formed.
    """
    forecasts = {(f.trial_id, f.method_id): f for f in read_forecasts(run_id)}
    commitments = {(c.trial_id, c.method_id): c for c in read_commitments(run_id)}
    reveals = {(r.trial_id, r.method_id): r for r in read_reveals(run_id)}

    checked = 0
    failures: list[dict[str, str]] = []

    for key, commitment in commitments.items():
        trial_id, method_id = key
        reveal = reveals.get(key)
        if reveal is None:
            failures.append(
                {"trial_id": trial_id, "method_id": method_id, "reason": "no selection reveal"}
            )
            continue
        forecast = forecasts.get(key)
        if forecast is None:
            failures.append(
                {"trial_id": trial_id, "method_id": method_id, "reason": "no forecast record"}
            )
            continue
        if reveal.commitment_hash != commitment.commitment_hash:
            failures.append(
                {
                    "trial_id": trial_id,
                    "method_id": method_id,
                    "reason": "reveal cites a different commitment hash",
                }
            )
            continue

        recomputed = commitment_hash(forecast_payload(forecast), salt_from_hex(reveal.salt_hex))
        checked += 1
        if recomputed != commitment.commitment_hash:
            failures.append(
                {
                    "trial_id": trial_id,
                    "method_id": method_id,
                    "reason": "forecast does not match its commitment",
                }
            )

    unrevealed = sorted(set(commitments) - set(reveals))
    return {
        "run_id": run_id,
        "commitments": len(commitments),
        "forecasts": len(forecasts),
        "reveals": len(reveals),
        "checked": checked,
        "verified": not failures and checked == len(commitments) and checked > 0,
        "failures": failures,
        "unrevealed": [{"trial_id": t, "method_id": m} for t, m in unrevealed],
    }
