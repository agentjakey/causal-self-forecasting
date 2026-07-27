"""Tests for the scoring metrics and the scoring runner.

The runner is tested against hand-built artifacts written straight to disk. No model is
loaded: scoring reads forecasts, observations, and the resolution manifest, so a run can be
faked cheaply and deterministically, which lets these tests cover the no-op exclusion and the
grouped bootstrap without minutes of forward passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_self_forecasting.hashing import atomic_write_json, write_jsonl
from causal_self_forecasting.paths import (
    FORECASTS,
    OBSERVATION_RECORDS,
    RESOLUTION_MANIFEST,
    SCORES,
    TRIAL_MANIFEST,
    run_dir,
)
from causal_self_forecasting.schemas import (
    ForecastCandidate,
    ForecastRecord,
    Framing,
    Mechanism,
    ModelVariant,
    ObservationRecord,
    PublicDashboardRecord,
    ResultStatus,
    Split,
    StateCondition,
    TrialRecord,
)
from causal_self_forecasting.scoring import ScoringError, score_run
from causal_self_forecasting.scoring.metrics import (
    PairScore,
    brier_score,
    grouped_bootstrap_ci,
    log_loss,
    mae,
    rmse,
    sign_accuracy,
    top_effect_accuracy,
)

RUN_ID = "score-test"


def _pair(
    *,
    trial: str = "trial_00000",
    intervention: str = "trial_00000.opaque_00",
    group: str = "g0",
    mechanism: str = "residual_add",
    predicted: float = 0.0,
    observed: float = 0.0,
    flip_prob: float = 0.5,
    flip: bool = False,
    low: float = -1.0,
    high: float = 1.0,
) -> PairScore:
    return PairScore(
        trial_id=trial,
        method_id="m",
        intervention_id=intervention,
        group_id=group,
        split="test",
        mechanism=mechanism,
        is_noop=mechanism == "noop",
        predicted_delta=predicted,
        observed_delta=observed,
        predicted_flip_probability=flip_prob,
        observed_flip=flip,
        interval_low=low,
        interval_high=high,
    )


# ---------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------


def test_mae_and_rmse() -> None:
    scores = [_pair(predicted=1.0, observed=0.0), _pair(predicted=0.0, observed=3.0)]
    assert mae(scores) == pytest.approx((1.0 + 3.0) / 2)
    assert rmse(scores) == pytest.approx(((1.0 + 9.0) / 2) ** 0.5)


def test_sign_accuracy_counts_matching_signs() -> None:
    scores = [
        _pair(predicted=1.0, observed=2.0),
        _pair(predicted=-1.0, observed=-2.0),
        _pair(predicted=1.0, observed=-2.0),
    ]
    assert sign_accuracy(scores) == pytest.approx(2 / 3)


def test_sign_accuracy_treats_near_zero_as_its_own_class() -> None:
    assert _pair(predicted=0.0, observed=0.0).sign_correct
    assert not _pair(predicted=0.0, observed=1.0).sign_correct


def test_brier_and_log_loss() -> None:
    scores = [_pair(flip_prob=0.75, flip=True), _pair(flip_prob=0.25, flip=False)]
    assert brier_score(scores) == pytest.approx((0.0625 + 0.0625) / 2)
    assert log_loss(scores) > 0


def test_brier_is_bounded() -> None:
    assert _pair(flip_prob=1.0, flip=False).flip_brier == pytest.approx(1.0)
    assert _pair(flip_prob=1.0, flip=True).flip_brier == pytest.approx(0.0)


def test_metrics_reject_empty_input() -> None:
    for metric in (mae, rmse, sign_accuracy, brier_score, log_loss):
        with pytest.raises(ValueError):
            metric([])


def test_interval_coverage_and_width() -> None:
    covered = _pair(observed=0.5, low=-1.0, high=1.0)
    assert covered.interval_covered
    assert covered.interval_width == pytest.approx(2.0)
    assert not _pair(observed=5.0, low=-1.0, high=1.0).interval_covered


def test_top_effect_accuracy_picks_largest_absolute_effect() -> None:
    trial = [
        _pair(intervention="a", predicted=2.0, observed=0.1),
        _pair(intervention="b", predicted=0.1, observed=3.0),
    ]
    # Predicted top is a, observed top is b: a miss.
    assert top_effect_accuracy([trial]) == pytest.approx(0.0)
    trial2 = [
        _pair(intervention="a", predicted=2.0, observed=3.0),
        _pair(intervention="b", predicted=0.1, observed=0.2),
    ]
    assert top_effect_accuracy([trial2]) == pytest.approx(1.0)


def test_top_effect_accuracy_needs_multiple_candidates() -> None:
    with pytest.raises(ValueError):
        top_effect_accuracy([[_pair()]])


def test_grouped_bootstrap_collapses_to_point_for_one_group() -> None:
    scores = [_pair(group="g0", predicted=1.0, observed=0.0) for _ in range(5)]
    low, high = grouped_bootstrap_ci(scores, mae)
    assert low == high == pytest.approx(1.0)


def test_grouped_bootstrap_widens_with_multiple_groups() -> None:
    scores = [
        _pair(group="g0", predicted=0.0, observed=0.0),
        _pair(group="g1", predicted=0.0, observed=4.0),
    ]
    low, high = grouped_bootstrap_ci(scores, mae, resamples=500, seed=1)
    assert low < high
    assert low >= 0.0


def test_grouped_bootstrap_is_reproducible() -> None:
    scores = [_pair(group=f"g{i}", observed=float(i)) for i in range(6)]
    first = grouped_bootstrap_ci(scores, mae, seed=7)
    second = grouped_bootstrap_ci(scores, mae, seed=7)
    assert first == second


# ---------------------------------------------------------------------------
# score_run against hand-built artifacts
# ---------------------------------------------------------------------------


def _trial(trial_id: str, group_id: str, split: Split = Split.TEST) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        variant_id=f"{group_id}.neutral_a",
        item_id=group_id,
        group_id=group_id,
        model_variant=ModelVariant.CLEAN,
        framing=Framing.NEUTRAL,
        split=split,
        state_id=f"{group_id}.neutral_a.L2",
        clean_logits={"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
        clean_margin=1.0,
        clean_predicted_label="A",
        candidate_set_hash="sha256:" + "0" * 64,
    )


def _forecast(
    trial_id: str, mechanisms: dict[str, str], method: str = "constant"
) -> ForecastRecord:
    return ForecastRecord(
        trial_id=trial_id,
        method_id=method,
        candidate_forecasts=[
            ForecastCandidate(
                intervention_id=intervention,
                delta_margin_mean=0.0,
                delta_margin_q05=-1.0,
                delta_margin_q95=1.0,
                p_answer_flip=0.5,
                p_bias_suppressed=0.5,
            )
            for intervention in mechanisms
        ],
        p_hidden_bias_active=0.5,
        state_condition=StateCondition.NONE,
    )


def _observation(
    trial_id: str, intervention: str, mechanism: str, delta: float
) -> ObservationRecord:
    post_margin = 1.0 + delta
    return ObservationRecord(
        trial_id=trial_id,
        intervention_id=intervention,
        mechanism=Mechanism(mechanism),
        clean_logits={"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
        post_logits={"A": post_margin, "B": 0.0, "C": 0.0, "D": 0.0},
        clean_margin=1.0,
        post_margin=post_margin,
        delta_margin=delta,
        clean_predicted_label="A",
        post_predicted_label="A",
        answer_flip=False,
        clean_entropy=0.1,
        post_entropy=0.1,
        pre_norm=10.0,
        post_norm=10.0,
    )


def _write_scorable_run(
    directory_holder: Path,
    *,
    classification: str = "ground_truth_resolution",
    fixture_only: bool = True,
    commitments_verified: bool = False,
    include_noop: bool = True,
) -> None:
    """Write a two-item, four-candidate run to disk, ready to score."""
    trials = [_trial("trial_00000", "g0"), _trial("trial_00001", "g1")]

    mechanisms = {
        "opaque_00": "residual_add",
        "opaque_01": "residual_add",
        "opaque_02": "random_add",
    }
    if include_noop:
        mechanisms["opaque_03"] = "noop"

    forecasts = []
    observations = []
    for trial in trials:
        per_trial = {f"{trial.trial_id}.{name}": mech for name, mech in mechanisms.items()}
        forecasts.append(_forecast(trial.trial_id, per_trial))
        for intervention, mech in per_trial.items():
            delta = 0.0 if mech == "noop" else (2.0 if "00" in intervention else -0.5)
            observations.append(_observation(trial.trial_id, intervention, mech, delta))

    directory = run_dir(RUN_ID)
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / TRIAL_MANIFEST, trials)
    write_jsonl(directory / FORECASTS, forecasts)
    write_jsonl(directory / OBSERVATION_RECORDS, observations)
    atomic_write_json(
        directory / RESOLUTION_MANIFEST,
        {
            "classification": classification,
            "fixture_only": fixture_only,
            "commitments_verified": commitments_verified,
            "counts": {"failures": 0},
        },
    )


def test_score_run_excludes_noop_from_headline(isolated_runs: Path) -> None:
    _write_scorable_run(isolated_runs)
    report = score_run(RUN_ID)
    method = report["methods"]["constant"]
    # Four candidates per trial, two trials: 8 pairs total, 6 non-no-op.
    assert method["all_candidates_including_noop"]["n"] == 8
    assert method["headline_excludes_noop"]["n"] == 6
    assert method["counts"]["noop_pairs"] == 2


def test_score_run_reports_all_candidate_metrics(isolated_runs: Path) -> None:
    _write_scorable_run(isolated_runs)
    method = score_run(RUN_ID)["methods"]["constant"]
    assert "mae" in method["all_candidates_including_noop"]
    assert "mae" in method["headline_excludes_noop"]
    # The no-op-inclusive MAE differs from the headline because no-ops are perfectly predicted.
    assert method["all_candidates_including_noop"]["mae"] != method["headline_excludes_noop"]["mae"]


def test_score_run_ranking_uses_complete_trials(isolated_runs: Path) -> None:
    _write_scorable_run(isolated_runs)
    method = score_run(RUN_ID)["methods"]["constant"]
    # Both trials have all four candidates observed, so both contribute to ranking.
    assert method["candidate_ranking"]["n_trials"] == 2
    assert "top_effect_accuracy" in method["candidate_ranking"]


def test_score_run_reports_grouped_intervals(isolated_runs: Path) -> None:
    _write_scorable_run(isolated_runs)
    block = score_run(RUN_ID)["methods"]["constant"]["headline_excludes_noop"]
    assert block["n_groups"] == 2
    assert len(block["mae_ci"]) == 2
    assert block["mae_ci"][0] <= block["mae"] <= block["mae_ci"][1]


def test_score_run_refuses_unverified_scientific_run(isolated_runs: Path) -> None:
    _write_scorable_run(
        isolated_runs,
        classification="scientific_forecast_resolution",
        fixture_only=False,
        commitments_verified=False,
    )
    with pytest.raises(ScoringError, match="did not verify"):
        score_run(RUN_ID)


def test_score_run_marks_fixture_non_scientific(isolated_runs: Path) -> None:
    _write_scorable_run(isolated_runs)
    report = score_run(RUN_ID)
    assert report["scored_as_scientific"] is False
    assert report["scientific_result"] is False
    assert report["fixture_only"] is True


def test_score_run_requires_resolution(isolated_runs: Path) -> None:
    directory = run_dir(RUN_ID)
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / TRIAL_MANIFEST, [_trial("trial_00000", "g0")])
    with pytest.raises(ScoringError, match="has not been resolved"):
        score_run(RUN_ID)


def test_score_run_refuses_output_collision(isolated_runs: Path) -> None:
    _write_scorable_run(isolated_runs)
    score_run(RUN_ID)
    with pytest.raises(ScoringError, match="already exists"):
        score_run(RUN_ID)
    # Force rescore succeeds.
    assert score_run(RUN_ID, force=True)["run_id"] == RUN_ID


def test_score_run_does_not_write_a_public_result(isolated_runs: Path) -> None:
    _write_scorable_run(isolated_runs)
    score_run(RUN_ID)
    assert (run_dir(RUN_ID) / SCORES).exists()
    # A scored run cannot be dressed up as a public export.
    with pytest.raises(Exception, match="did not verify"):
        PublicDashboardRecord(
            run_id=RUN_ID,
            status=ResultStatus.PRELIMINARY,
            commitments_verified=False,
        )
