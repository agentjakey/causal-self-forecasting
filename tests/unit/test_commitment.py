"""Tests for the commit, select, reveal, verify protocol.

The properties under test are the ones the protocol claims: a forecast cannot be committed
after a selection seed exists, a seed cannot be written before commitments exist, selection
is deterministic, and an edited forecast fails verification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_self_forecasting.hashing import read_jsonl, write_jsonl
from causal_self_forecasting.paths import FORECASTS, run_dir
from causal_self_forecasting.schemas import (
    CandidateSet,
    ForecastCandidate,
    ForecastRecord,
    InterventionSpec,
    Mechanism,
)
from causal_self_forecasting.trials.commitment import (
    ProtocolOrderError,
    commit_forecast,
    read_commitments,
    reveal_selection,
    select_candidate_index,
    verify_run_commitments,
    write_selection_seed,
)

RUN_ID = "test-run"
SEED_HEX = "a1b2c3d4e5f6a7b8"


def _candidate(index: int, mechanism: Mechanism = Mechanism.RESIDUAL_ADD) -> InterventionSpec:
    return InterventionSpec(
        intervention_id=f"trial_00001.opaque_{index:02d}",
        mechanism=mechanism,
        mechanism_version="1.0",
        layer=2,
        position_index=-1,
        strength=1.0 if mechanism is not Mechanism.NOOP else 0.0,
        direction_id="d1" if mechanism is not Mechanism.NOOP else None,
    )


def _candidate_set(trial_id: str = "trial_00001") -> CandidateSet:
    return CandidateSet(
        trial_id=trial_id,
        candidates=[_candidate(index) for index in range(4)],
        order_seed=42,
    )


def _forecast(trial_id: str = "trial_00001", delta: float = -0.5) -> ForecastRecord:
    return ForecastRecord(
        trial_id=trial_id,
        method_id="state_mlp",
        candidate_forecasts=[
            ForecastCandidate(
                intervention_id=f"{trial_id}.opaque_{index:02d}",
                delta_margin_mean=delta,
                delta_margin_q05=delta - 0.5,
                delta_margin_q95=delta + 0.5,
                p_answer_flip=0.3,
                p_bias_suppressed=0.6,
            )
            for index in range(4)
        ],
        p_hidden_bias_active=0.7,
    )


def test_commit_writes_forecast_and_commitment(isolated_runs: Path) -> None:
    commitment = commit_forecast(RUN_ID, _forecast())
    assert commitment.commitment_hash.startswith("sha256:")
    assert (run_dir(RUN_ID) / FORECASTS).exists()
    assert len(read_commitments(RUN_ID)) == 1


def test_commit_does_not_write_the_salt_into_the_commitment(isolated_runs: Path) -> None:
    """A commitment that carried its own salt would not hide anything."""
    commitment = commit_forecast(RUN_ID, _forecast())
    assert "salt" not in commitment.model_dump(mode="json")


def test_seed_cannot_be_written_before_any_commitment(isolated_runs: Path) -> None:
    with pytest.raises(ProtocolOrderError, match="no commitments yet"):
        write_selection_seed(RUN_ID, SEED_HEX)


def test_commit_is_refused_once_a_seed_exists(isolated_runs: Path) -> None:
    """The core sequencing guarantee: no forecasting after the selection is knowable."""
    commit_forecast(RUN_ID, _forecast("trial_00001"))
    write_selection_seed(RUN_ID, SEED_HEX)

    with pytest.raises(ProtocolOrderError, match="already has a selection seed"):
        commit_forecast(RUN_ID, _forecast("trial_00002"))


def test_seed_cannot_be_silently_rewritten(isolated_runs: Path) -> None:
    """Rewriting a seed would allow reselecting until a convenient candidate came up."""
    commit_forecast(RUN_ID, _forecast())
    write_selection_seed(RUN_ID, SEED_HEX)
    with pytest.raises(ProtocolOrderError, match="already has a selection seed"):
        write_selection_seed(RUN_ID, "ffffffffffffffff")


def test_selection_is_deterministic() -> None:
    candidate_set = _candidate_set()
    first = select_candidate_index(SEED_HEX, candidate_set)
    second = select_candidate_index(SEED_HEX, candidate_set)
    assert first == second


def test_selection_is_in_range() -> None:
    candidate_set = _candidate_set()
    assert 0 <= select_candidate_index(SEED_HEX, candidate_set) < 4


def test_selection_depends_on_the_trial_id() -> None:
    """Per-trial derivation, so one trial can be recomputed without replaying the run."""
    picks = {
        select_candidate_index(SEED_HEX, _candidate_set(f"trial_{index:05d}"))
        for index in range(40)
    }
    assert len(picks) > 1


def test_selection_depends_on_the_seed() -> None:
    candidate_set = _candidate_set()
    picks = {select_candidate_index(f"seed{index:04d}", candidate_set) for index in range(40)}
    assert len(picks) > 1


def test_reveal_verifies_an_untouched_forecast(isolated_runs: Path) -> None:
    forecast = _forecast()
    commitment = commit_forecast(RUN_ID, forecast)
    write_selection_seed(RUN_ID, SEED_HEX)

    reveal = reveal_selection(RUN_ID, _candidate_set(), commitment, forecast, SEED_HEX)
    assert reveal.verified
    assert reveal.no_selection is False
    assert reveal.selected_intervention_id is not None
    assert reveal.selected_intervention_id.startswith("trial_00001.opaque_")


def test_reveal_does_not_expose_the_raw_seed(isolated_runs: Path) -> None:
    forecast = _forecast()
    commitment = commit_forecast(RUN_ID, forecast)
    write_selection_seed(RUN_ID, SEED_HEX)
    reveal = reveal_selection(RUN_ID, _candidate_set(), commitment, forecast, SEED_HEX)
    assert SEED_HEX not in reveal.model_dump_json()
    assert reveal.selection_seed_hash is not None
    assert reveal.selection_seed_hash.startswith("sha256:")


def test_verify_run_passes_for_an_honest_run(isolated_runs: Path) -> None:
    forecast = _forecast()
    commitment = commit_forecast(RUN_ID, forecast)
    write_selection_seed(RUN_ID, SEED_HEX)
    reveal_selection(RUN_ID, _candidate_set(), commitment, forecast, SEED_HEX)

    report = verify_run_commitments(RUN_ID)
    assert report["verified"]
    assert report["checked"] == 1
    assert report["failures"] == []


def test_verify_run_detects_an_edited_forecast(isolated_runs: Path) -> None:
    """The scenario the protocol exists for: results seen, then the forecast quietly fixed."""
    forecast = _forecast(delta=-0.5)
    commitment = commit_forecast(RUN_ID, forecast)
    write_selection_seed(RUN_ID, SEED_HEX)
    reveal_selection(RUN_ID, _candidate_set(), commitment, forecast, SEED_HEX)
    assert verify_run_commitments(RUN_ID)["verified"]

    # Rewrite the forecast on disk to a value that matches the outcome better.
    rows = list(read_jsonl(run_dir(RUN_ID) / FORECASTS))
    rows[0]["candidate_forecasts"][0]["delta_margin_mean"] = -0.9
    write_jsonl(run_dir(RUN_ID) / FORECASTS, rows)

    report = verify_run_commitments(RUN_ID)
    assert not report["verified"]
    assert report["failures"][0]["reason"] == "forecast does not match its commitment"


def test_verify_run_flags_a_commitment_with_no_reveal(isolated_runs: Path) -> None:
    commit_forecast(RUN_ID, _forecast())
    report = verify_run_commitments(RUN_ID)
    assert not report["verified"]
    assert report["unrevealed"] == [
        {
            "trial_id": "trial_00001",
            "method_id": "state_mlp",
            "state_condition": "true",
            "condition_index": "0",
        }
    ]


def test_verify_run_is_not_vacuously_true_for_an_empty_run(isolated_runs: Path) -> None:
    """An empty run must not report `verified: true`; nothing was checked."""
    report = verify_run_commitments("nonexistent-run")
    assert not report["verified"]
    assert report["checked"] == 0


def test_reveal_without_a_salt_is_refused(isolated_runs: Path) -> None:
    forecast = _forecast()
    commitment = commit_forecast(RUN_ID, forecast)
    write_selection_seed(RUN_ID, SEED_HEX)

    from causal_self_forecasting.trials.commitment import _salt_path

    _salt_path(RUN_ID, forecast.trial_id, forecast.method_id).unlink()
    with pytest.raises(ProtocolOrderError, match="no salt found"):
        reveal_selection(RUN_ID, _candidate_set(), commitment, forecast, SEED_HEX)
