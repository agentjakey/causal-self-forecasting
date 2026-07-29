"""Tests for the hardened commitment protocol.

Three properties the state-dependence arm needs and the original protocol did not have: a key
that distinguishes twelve records per prompt, a reveal that can verify without selecting, and a
committer that refuses once an outcome exists.

The last one is the important one. Timestamps alone cannot catch "resolve, look, commit": the
sequence leaves a commitment that verifies perfectly and an ordering that reads correctly. The
only defence is refusing to commit at all while an outcome artifact is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_self_forecasting.hashing import append_jsonl, atomic_write_json
from causal_self_forecasting.paths import (
    OBSERVATION_RECORDS,
    RESOLUTION_MANIFEST,
    SCORES,
    STATE_AUDIT_OBSERVATIONS,
    run_dir,
)
from causal_self_forecasting.schemas import (
    ForecastCandidate,
    ForecastRecord,
    SelectionReveal,
    StateCondition,
)
from causal_self_forecasting.trials.commitment import (
    ProtocolOrderError,
    commit_forecast,
    commitment_key,
    existing_outcome_artifacts,
    read_commitments,
    record_key,
    reveal_without_selection,
    verify_commitment_ordering,
    verify_run_commitments,
)

RUN_ID = "harden-run"


def _forecast(
    method_id: str = "state_bilinear_ridge",
    condition: StateCondition = StateCondition.TRUE,
    condition_index: int = 0,
    trial_id: str = "sa_00000",
    mean: float = 0.25,
) -> ForecastRecord:
    return ForecastRecord(
        trial_id=trial_id,
        method_id=method_id,
        candidate_forecasts=[
            ForecastCandidate(
                intervention_id=f"{trial_id}.opaque_{index:02d}",
                delta_margin_mean=mean,
                delta_margin_q05=mean - 1.0,
                delta_margin_q95=mean + 1.0,
                p_answer_flip=0.1,
                p_bias_suppressed=0.5,
            )
            for index in range(3)
        ],
        p_hidden_bias_active=0.5,
        state_condition=condition,
        condition_index=condition_index,
    )


# ---------------------------------------------------------------------------
# The four-part key
# ---------------------------------------------------------------------------


def test_the_key_carries_all_four_components() -> None:
    assert commitment_key("t", "m", StateCondition.SHUFFLED, 7) == ("t", "m", "shuffled", 7)


def test_twelve_records_for_one_prompt_and_method_do_not_collide(isolated_runs: Path) -> None:
    """The exact shape the arm commits: one true, one matched, ten permutations."""
    plans = [
        (StateCondition.TRUE, 0),
        (StateCondition.WRONG_EXAMPLE, 0),
        *[(StateCondition.SHUFFLED, index) for index in range(10)],
    ]
    for condition, index in plans:
        commit_forecast(RUN_ID, _forecast(condition=condition, condition_index=index))

    commitments = read_commitments(RUN_ID)
    assert len(commitments) == 12
    keys = {record_key(commitment) for commitment in commitments}
    assert len(keys) == 12


def test_every_commitment_gets_its_own_salt_file(isolated_runs: Path) -> None:
    for index in range(10):
        commit_forecast(RUN_ID, _forecast(condition=StateCondition.SHUFFLED, condition_index=index))
    salts = sorted((run_dir(RUN_ID) / "private_payloads" / "salts").glob("*.salt"))
    assert len(salts) == 10
    assert len({path.read_text(encoding="utf-8") for path in salts}) == 10


def test_committing_the_same_key_twice_is_refused(isolated_runs: Path) -> None:
    commit_forecast(RUN_ID, _forecast())
    with pytest.raises(ProtocolOrderError, match="already has a commitment"):
        commit_forecast(RUN_ID, _forecast(mean=9.0))


def test_the_same_method_under_two_conditions_is_not_a_duplicate(isolated_runs: Path) -> None:
    commit_forecast(RUN_ID, _forecast(condition=StateCondition.TRUE))
    commit_forecast(RUN_ID, _forecast(condition=StateCondition.WRONG_EXAMPLE))
    assert len(read_commitments(RUN_ID)) == 2


# ---------------------------------------------------------------------------
# Refusing to commit once an outcome exists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "artifact", [OBSERVATION_RECORDS, STATE_AUDIT_OBSERVATIONS, RESOLUTION_MANIFEST, SCORES]
)
def test_committing_is_refused_once_an_outcome_artifact_exists(
    isolated_runs: Path, artifact: str
) -> None:
    directory = run_dir(RUN_ID)
    directory.mkdir(parents=True, exist_ok=True)
    if artifact.endswith(".jsonl"):
        append_jsonl(directory / artifact, {"anything": 1})
    else:
        atomic_write_json(directory / artifact, {"anything": 1})

    with pytest.raises(ProtocolOrderError, match="is not a forecast"):
        commit_forecast(RUN_ID, _forecast())


def test_existing_outcome_artifacts_reports_what_is_there(isolated_runs: Path) -> None:
    directory = run_dir(RUN_ID)
    directory.mkdir(parents=True, exist_ok=True)
    assert existing_outcome_artifacts(RUN_ID) == []
    append_jsonl(directory / STATE_AUDIT_OBSERVATIONS, {"anything": 1})
    assert existing_outcome_artifacts(RUN_ID) == [STATE_AUDIT_OBSERVATIONS]


def test_ordering_reports_no_outcomes_at_the_commitment_checkpoint(isolated_runs: Path) -> None:
    commit_forecast(RUN_ID, _forecast())
    report = verify_commitment_ordering(RUN_ID)
    assert report["final_test_outcomes_exist"] is False
    assert report["outcome_artifacts_present"] == []
    assert report["ordering_valid"] is True
    assert report["commitments"] == 1
    assert report["latest_committed_at"] is not None
    assert report["earliest_observed_at"] is None


# ---------------------------------------------------------------------------
# No-selection reveal
# ---------------------------------------------------------------------------


def test_a_no_selection_reveal_verifies_without_choosing_a_candidate(isolated_runs: Path) -> None:
    forecast = _forecast()
    commitment = commit_forecast(RUN_ID, forecast)

    reveal = reveal_without_selection(RUN_ID, commitment, forecast)
    assert reveal.verified is True
    assert reveal.no_selection is True
    assert reveal.selected_intervention_id is None
    assert reveal.selected_index is None
    assert reveal.selection_seed_hash is None

    report = verify_run_commitments(RUN_ID)
    assert report["verified"] is True
    assert report["checked"] == 1


def test_a_no_selection_reveal_detects_an_edited_forecast(isolated_runs: Path) -> None:
    forecast = _forecast()
    commitment = commit_forecast(RUN_ID, forecast)
    tampered = _forecast(mean=99.0)
    reveal = reveal_without_selection(RUN_ID, commitment, tampered)
    assert reveal.verified is False


def test_all_twelve_conditions_verify_through_no_selection_reveals(isolated_runs: Path) -> None:
    plans = [
        (StateCondition.TRUE, 0),
        (StateCondition.WRONG_EXAMPLE, 0),
        *[(StateCondition.SHUFFLED, index) for index in range(10)],
    ]
    for condition, index in plans:
        forecast = _forecast(condition=condition, condition_index=index)
        commitment = commit_forecast(RUN_ID, forecast)
        reveal_without_selection(RUN_ID, commitment, forecast)

    report = verify_run_commitments(RUN_ID)
    assert report["commitments"] == 12
    assert report["checked"] == 12
    assert report["verified"] is True
    assert report["failures"] == []


def test_a_no_selection_reveal_may_not_name_a_candidate() -> None:
    with pytest.raises(ValueError, match="must not name a selected candidate"):
        SelectionReveal(
            trial_id="t",
            method_id="m",
            no_selection=True,
            selected_intervention_id="t.opaque_00",
            selected_index=0,
            salt_hex="ab" * 16,
            commitment_hash="sha256:" + "0" * 64,
            verified=True,
        )


def test_a_selecting_reveal_still_requires_its_selection() -> None:
    with pytest.raises(ValueError, match="requires both"):
        SelectionReveal(
            trial_id="t",
            method_id="m",
            selection_seed_hash="sha256:" + "1" * 64,
            salt_hex="ab" * 16,
            commitment_hash="sha256:" + "0" * 64,
            verified=True,
        )
