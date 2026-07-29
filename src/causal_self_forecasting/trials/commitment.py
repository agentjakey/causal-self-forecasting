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
    OBSERVATION_RECORDS,
    RESOLUTION_MANIFEST,
    SCORES,
    SELECTION_REVEALS,
    STATE_AUDIT_OBSERVATIONS,
    private_dir,
    run_dir,
)
from ..reproducibility import derive_seed
from ..schemas import (
    CandidateSet,
    ForecastCommitment,
    ForecastRecord,
    SelectionReveal,
    StateCondition,
)

SELECTION_SEED_FILENAME = "selection_seed.txt"

# Artifacts whose existence means an outcome is already known. Committing a forecast after any
# of them exists is not a forecast, whatever the timestamps say, so the committer refuses rather
# than recording a commitment that would still verify.
OUTCOME_ARTIFACTS = (
    OBSERVATION_RECORDS,
    STATE_AUDIT_OBSERVATIONS,
    RESOLUTION_MANIFEST,
    SCORES,
)

# The commitment key. Four components, because the state-dependence arm commits twelve records
# per prompt for one method and they differ only in the state condition and its index.
type CommitmentKey = tuple[str, str, str, int]


class ProtocolOrderError(RuntimeError):
    """Raised when the commit/select/reveal steps are attempted out of order."""


class CommitmentVerificationError(RuntimeError):
    """Raised when a revealed commitment does not recompute."""


def commitment_key(
    trial_id: str,
    method_id: str,
    state_condition: StateCondition | str = StateCondition.TRUE,
    condition_index: int = 0,
) -> CommitmentKey:
    """The identity of one commitment.

    One function, so the committer, the duplicate check, the salt path, and the verifier cannot
    drift into disagreeing about what counts as the same commitment.
    """
    condition = (
        state_condition.value if isinstance(state_condition, StateCondition) else state_condition
    )
    return (trial_id, method_id, condition, int(condition_index))


def record_key(record: ForecastRecord | ForecastCommitment | SelectionReveal) -> CommitmentKey:
    return commitment_key(
        record.trial_id, record.method_id, record.state_condition, record.condition_index
    )


def _salt_path(
    run_id: str,
    trial_id: str,
    method_id: str,
    state_condition: StateCondition | str = StateCondition.TRUE,
    condition_index: int = 0,
) -> Path:
    """Where one commitment's salt lives.

    The filename carries all four key components. With only trial and method in the path, the
    ten permutation commitments for a prompt would overwrite each other's salt and the
    overwritten ones would become unverifiable.
    """
    trial, method, condition, index = commitment_key(
        trial_id, method_id, state_condition, condition_index
    )
    return private_dir(run_id) / "salts" / f"{trial}.{method}.{condition}.{index:02d}.salt"


def existing_outcome_artifacts(run_id: str) -> list[str]:
    """Outcome artifacts already present in a run directory."""
    directory = run_dir(run_id)
    return [name for name in OUTCOME_ARTIFACTS if (directory / name).exists()]


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

    # The sequence "resolve, look, commit" leaves timestamps that all read correctly and a
    # commitment that verifies perfectly. The only way to catch it is to refuse to commit once
    # an outcome artifact exists at all.
    outcomes = existing_outcome_artifacts(run_id)
    if outcomes:
        raise ProtocolOrderError(
            f"run {run_id} already holds outcome artifacts {outcomes}; a forecast committed after "
            "an outcome exists is not a forecast. Generate the trials into a fresh run directory "
            "and commit there, and record the discarded run in docs/experiment_log.md."
        )

    key = record_key(forecast)
    if key in {record_key(existing) for existing in read_commitments(run_id)}:
        raise ProtocolOrderError(
            f"run {run_id} already has a commitment for {key}; committing twice under one key "
            "would leave the second one unverifiable"
        )

    salt = new_salt()
    digest = commitment_hash(forecast_payload(forecast), salt)

    directory = run_dir(run_id)
    append_jsonl(directory / FORECASTS, forecast)

    commitment = ForecastCommitment(
        trial_id=forecast.trial_id,
        method_id=forecast.method_id,
        state_condition=forecast.state_condition,
        condition_index=forecast.condition_index,
        commitment_hash=digest,
        forecast_ref=FORECASTS,
    )
    append_jsonl(directory / FORECAST_COMMITMENTS, commitment)

    salt_file = _salt_path(
        run_id,
        forecast.trial_id,
        forecast.method_id,
        forecast.state_condition,
        forecast.condition_index,
    )
    salt_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(salt_file, salt_to_hex(salt))

    info(
        "committed forecast",
        run_id=run_id,
        trial_id=forecast.trial_id,
        method_id=forecast.method_id,
        state_condition=forecast.state_condition.value,
        condition_index=forecast.condition_index,
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
    salt = _read_salt(run_id, forecast)
    verified = verify_commitment(forecast_payload(forecast), salt, commitment.commitment_hash)

    reveal = SelectionReveal(
        trial_id=forecast.trial_id,
        method_id=forecast.method_id,
        state_condition=forecast.state_condition,
        condition_index=forecast.condition_index,
        selection_seed_hash=sha256_hex(seed_hex.encode("utf-8")),
        selected_intervention_id=selected.intervention_id,
        selected_index=index,
        salt_hex=salt_to_hex(salt),
        commitment_hash=commitment.commitment_hash,
        verified=verified,
    )
    append_jsonl(run_dir(run_id) / SELECTION_REVEALS, reveal)
    return reveal


def _read_salt(run_id: str, forecast: ForecastRecord) -> bytes:
    salt_file = _salt_path(
        run_id,
        forecast.trial_id,
        forecast.method_id,
        forecast.state_condition,
        forecast.condition_index,
    )
    if not salt_file.exists():
        raise ProtocolOrderError(
            f"no salt found for {record_key(forecast)} at {salt_file}; the forecast cannot be "
            "revealed"
        )
    return salt_from_hex(salt_file.read_text(encoding="utf-8"))


def reveal_without_selection(
    run_id: str,
    commitment: ForecastCommitment,
    forecast: ForecastRecord,
) -> SelectionReveal:
    """Disclose the salt and verify the commitment without selecting a candidate.

    The all-candidate path. Every candidate in the set is resolved, so there is nothing for a
    seed to choose, and the blinding rests entirely on ordering: the commitment existed before
    any outcome artifact did. A verification failure is recorded rather than raised, for the same
    reason it is in `reveal_selection`.
    """
    salt = _read_salt(run_id, forecast)
    verified = verify_commitment(forecast_payload(forecast), salt, commitment.commitment_hash)

    reveal = SelectionReveal(
        trial_id=forecast.trial_id,
        method_id=forecast.method_id,
        state_condition=forecast.state_condition,
        condition_index=forecast.condition_index,
        no_selection=True,
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
    forecasts = {record_key(f): f for f in read_forecasts(run_id)}
    commitments = {record_key(c): c for c in read_commitments(run_id)}
    reveals = {record_key(r): r for r in read_reveals(run_id)}

    checked = 0
    failures: list[dict[str, str]] = []

    def _fail(key: CommitmentKey, reason: str) -> None:
        trial_id, method_id, condition, index = key
        failures.append(
            {
                "trial_id": trial_id,
                "method_id": method_id,
                "state_condition": condition,
                "condition_index": str(index),
                "reason": reason,
            }
        )

    for key, commitment in commitments.items():
        reveal = reveals.get(key)
        if reveal is None:
            _fail(key, "no reveal")
            continue
        forecast = forecasts.get(key)
        if forecast is None:
            _fail(key, "no forecast record")
            continue
        if reveal.commitment_hash != commitment.commitment_hash:
            _fail(key, "reveal cites a different commitment hash")
            continue

        recomputed = commitment_hash(forecast_payload(forecast), salt_from_hex(reveal.salt_hex))
        checked += 1
        if recomputed != commitment.commitment_hash:
            _fail(key, "forecast does not match its commitment")

    unrevealed = sorted(set(commitments) - set(reveals))
    return {
        "run_id": run_id,
        "commitments": len(commitments),
        "forecasts": len(forecasts),
        "reveals": len(reveals),
        "checked": checked,
        "verified": not failures and checked == len(commitments) and checked > 0,
        "failures": failures,
        "unrevealed": [
            {
                "trial_id": t,
                "method_id": m,
                "state_condition": c,
                "condition_index": str(i),
            }
            for t, m, c, i in unrevealed
        ],
    }


def verify_commitment_ordering(run_id: str) -> dict[str, Any]:
    """Check from artifacts, not from prose, that every forecast predates every outcome.

    Three things, all recomputable by a third party: no outcome artifact exists while the run is
    still at the commitment checkpoint; if outcomes do exist, every `committed_at` precedes every
    `observed_at`; and no commitment key is duplicated.
    """
    directory = run_dir(run_id)
    commitments = read_commitments(run_id)
    outcomes = existing_outcome_artifacts(run_id)
    failures: list[str] = []

    keys = [record_key(commitment) for commitment in commitments]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        failures.append(f"duplicate commitment keys: {duplicates[:5]}")

    latest_commit = max((c.committed_at for c in commitments), default=None)
    earliest_outcome = None

    observations_path = directory / STATE_AUDIT_OBSERVATIONS
    if observations_path.exists():
        from ..schemas import StateAuditObservationRecord

        stamps = [
            StateAuditObservationRecord.model_validate(row).observed_at
            for row in read_jsonl(observations_path)
        ]
        earliest_outcome = min(stamps, default=None)

    if (
        earliest_outcome is not None
        and latest_commit is not None
        and latest_commit >= earliest_outcome
    ):
        failures.append(
            f"the last commitment at {latest_commit.isoformat()} does not precede the first "
            f"outcome at {earliest_outcome.isoformat()}"
        )

    return {
        "run_id": run_id,
        "commitments": len(commitments),
        "outcome_artifacts_present": outcomes,
        "final_test_outcomes_exist": bool(outcomes),
        "latest_committed_at": latest_commit.isoformat() if latest_commit else None,
        "earliest_observed_at": earliest_outcome.isoformat() if earliest_outcome else None,
        "ordering_valid": not failures,
        "failures": failures,
    }
