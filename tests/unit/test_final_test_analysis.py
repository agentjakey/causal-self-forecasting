"""Unit tests for the final-test guards, scoring, and preregistered analysis.

No model and no run execution. The scoring and analysis path is entirely model-free by design, so
almost all of it can be tested against hand-built forecasts and outcomes, which is also the right
level for the properties that matter: prompt-first aggregation, the paired bootstrap, the decision
rule, and the refusals that protect the blinding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_self_forecasting.hashing import append_jsonl, atomic_write_json, write_jsonl
from causal_self_forecasting.paths import (
    SELECTION_REVEALS,
    STATE_AUDIT_OBSERVATIONS,
    run_dir,
)
from causal_self_forecasting.schemas import (
    ForecastCandidate,
    ForecastRecord,
    MethodConditionSummary,
    PairedComparison,
    PromptRole,
    StateAuditObservationRecord,
    StateCondition,
)
from causal_self_forecasting.scoring.metrics import (
    paired_grouped_bootstrap,
    prompt_first_mean,
    spearman_correlation,
)
from causal_self_forecasting.state_audit.analyze import (
    BRIER_MIN_FLIPS,
    MATCHED_WRONG,
    TRUE_STATE,
    VISIBLE,
    AnalysisError,
    compare,
    permutation_band,
    score_conditions,
    summarize_condition,
)
from causal_self_forecasting.state_audit.resolve import (
    EXPECTED_RECORDS_PER_PROMPT,
    FinalTestResolutionError,
    check_commitment_state,
    check_working_tree,
    reveal_all,
    verify_stored_targets,
)
from causal_self_forecasting.trials.commitment import (
    commit_forecast,
    read_reveals,
    verify_commitment_ordering,
    verify_run_commitments,
)

RUN_ID = "ft-run"
CLEAN = {"A": 3.0, "B": 1.0, "C": 0.5, "D": 0.0}
HASH = "sha256:" + "0" * 64

PROMPTS = 4
SIGNED = 16


def observation(prompt: int, index: int, delta: float, is_noop: bool = False):
    intervened = dict(CLEAN) | {"A": CLEAN["A"] + delta}
    return StateAuditObservationRecord(
        study_id="s",
        run_id=RUN_ID,
        trial_id=f"sa_{prompt:05d}",
        candidate_id=f"sa_{prompt:05d}.opaque_{index:02d}",
        group_id=f"g{prompt:04d}",
        variant_id=f"g{prompt:04d}.neutral_a",
        prompt_role=PromptRole.FINAL_TEST,
        clean_preferred_label="A",
        clean_logits=dict(CLEAN),
        intervened_logits=intervened,
        clean_top_margin=2.0,
        intervened_top_margin=2.0 + delta,
        delta_clean_top_margin=delta,
        answer_flip=(2.0 + delta) < 0.0,
        is_noop=is_noop,
        direction_ref="noop" if is_noop else f"bd1.{index:016x}",
        direction_vector_hash=None if is_noop else HASH,
        layer=13,
        norm_ratio=0.0 if is_noop else 0.02,
        global_alpha=0.0 if is_noop else 106.87158268272867,
        pre_norm=5000.0,
        post_norm=5001.0,
        delta_norm=0.0 if is_noop else 106.87158268272867,
        model_id="google/gemma-3-1b-it",
        model_revision="dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        prompt_manifest_hash=HASH,
        direction_family_hash=HASH,
        config_hash=HASH,
    )


def build_outcomes(flip_deltas: int = 0) -> list[StateAuditObservationRecord]:
    """One prompt set with a controllable number of realized flips.

    A flip needs the intervened top margin to go negative, so a delta below -2 flips.
    """
    rows: list[StateAuditObservationRecord] = []
    flips_left = flip_deltas
    for prompt in range(PROMPTS):
        for index in range(SIGNED):
            if flips_left > 0:
                delta = -3.0
                flips_left -= 1
            else:
                delta = 0.5 if index % 2 == 0 else -0.5
            rows.append(observation(prompt, index, delta))
        rows.append(observation(prompt, 99, 0.0, is_noop=True))
    return rows


def forecast_for(
    prompt: int,
    method_id: str,
    condition: StateCondition,
    index: int,
    predict: float,
) -> ForecastRecord:
    candidates = [
        ForecastCandidate(
            intervention_id=f"sa_{prompt:05d}.opaque_{candidate:02d}",
            delta_margin_mean=predict if candidate % 2 == 0 else -predict,
            delta_margin_q05=-5.0,
            delta_margin_q95=5.0,
            p_answer_flip=0.03,
            p_bias_suppressed=0.5,
        )
        for candidate in range(SIGNED)
    ]
    candidates.append(
        ForecastCandidate(
            intervention_id=f"sa_{prompt:05d}.opaque_99",
            delta_margin_mean=0.0,
            delta_margin_q05=-1.0,
            delta_margin_q95=1.0,
            p_answer_flip=0.03,
            p_bias_suppressed=0.5,
        )
    )
    return ForecastRecord(
        trial_id=f"sa_{prompt:05d}",
        method_id=method_id,
        candidate_forecasts=candidates,
        p_hidden_bias_active=0.5,
        state_condition=condition,
        condition_index=index,
    )


def build_forecasts(
    true_predict: float = 0.5,
    visible_predict: float = 0.9,
    wrong_predict: float = 0.9,
) -> list[ForecastRecord]:
    """The full 16-record shape per prompt, with tunable prediction quality per condition."""
    rows: list[ForecastRecord] = []
    for prompt in range(PROMPTS):
        rows.append(forecast_for(prompt, "constant", StateCondition.NONE, 0, 0.2))
        rows.append(forecast_for(prompt, "prompt_lexical", StateCondition.NONE, 0, 0.3))
        rows.append(forecast_for(prompt, "intervention_only_ridge", StateCondition.NONE, 0, 0.4))
        rows.append(
            forecast_for(
                prompt, "visible_information_ridge", StateCondition.NONE, 0, visible_predict
            )
        )
        rows.append(
            forecast_for(prompt, "state_bilinear_ridge", StateCondition.TRUE, 0, true_predict)
        )
        rows.append(
            forecast_for(
                prompt, "state_bilinear_ridge", StateCondition.WRONG_EXAMPLE, 0, wrong_predict
            )
        )
        for index in range(10):
            rows.append(
                forecast_for(prompt, "state_bilinear_ridge", StateCondition.SHUFFLED, index, 0.85)
            )
    return rows


# ---------------------------------------------------------------------------
# Prompt-first aggregation and the paired bootstrap
# ---------------------------------------------------------------------------


def test_prompt_first_mean_counts_each_group_once() -> None:
    assert prompt_first_mean({"a": 1.0, "b": 3.0}) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="zero groups"):
        prompt_first_mean({})


def test_prompt_first_aggregation_is_not_pair_pooling() -> None:
    """A prompt with many pairs must not dominate a prompt with few."""
    left = {"g0": 0.0, "g1": 1.0}
    assert prompt_first_mean(left) == pytest.approx(0.5)


def test_the_paired_bootstrap_uses_one_resample_for_both_sides() -> None:
    """Perfectly correlated errors give a zero-width interval; independent bootstraps would not."""
    left = {f"g{i}": float(i) + 1.0 for i in range(20)}
    right = {f"g{i}": float(i) for i in range(20)}
    result = paired_grouped_bootstrap(left, right, resamples=500, seed=1)
    assert result.point_estimate == pytest.approx(1.0)
    assert result.ci_low == pytest.approx(1.0)
    assert result.ci_high == pytest.approx(1.0)
    assert result.excludes_zero is True


def test_the_paired_bootstrap_is_deterministic_from_the_seed() -> None:
    left = {f"g{i}": float(i % 5) for i in range(30)}
    right = {f"g{i}": float((i * 7) % 5) for i in range(30)}
    first = paired_grouped_bootstrap(left, right, resamples=400, seed=7)
    second = paired_grouped_bootstrap(left, right, resamples=400, seed=7)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)

    # The point estimate is the observed paired difference, not a resampled quantity, so it must
    # not move with the seed at all. Only the interval endpoints may.
    other = paired_grouped_bootstrap(left, right, resamples=400, seed=8)
    assert other.point_estimate == pytest.approx(first.point_estimate)


def test_the_paired_bootstrap_reports_a_crossing_interval() -> None:
    left = {f"g{i}": 1.0 if i % 2 else -1.0 for i in range(20)}
    right = {f"g{i}": 0.0 for i in range(20)}
    result = paired_grouped_bootstrap(left, right, resamples=2000, seed=3)
    assert result.ci_low < 0.0 < result.ci_high
    assert result.excludes_zero is False
    assert result.sign == "no detected difference"


def test_the_paired_bootstrap_refuses_mismatched_groups() -> None:
    with pytest.raises(ValueError, match="same groups"):
        paired_grouped_bootstrap({"a": 1.0}, {"b": 1.0})


def test_spearman_is_none_when_undefined() -> None:
    assert spearman_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert spearman_correlation([1.0], [2.0]) is None
    assert spearman_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert spearman_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Scoring from sealed forecasts
# ---------------------------------------------------------------------------


def test_scoring_covers_every_committed_condition() -> None:
    scored = score_conditions(build_forecasts(), build_outcomes())
    assert len(scored) == EXPECTED_RECORDS_PER_PROMPT
    for entry in scored.values():
        assert len(entry.signed_by_prompt) == PROMPTS
        assert sum(len(v) for v in entry.signed_by_prompt.values()) == PROMPTS * SIGNED
        assert sum(len(v) for v in entry.noop_by_prompt.values()) == PROMPTS


def test_scoring_refuses_a_forecast_with_no_outcome() -> None:
    outcomes = [row for row in build_outcomes() if row.candidate_id != "sa_00000.opaque_00"]
    with pytest.raises(AnalysisError, match="has no outcome"):
        score_conditions(build_forecasts(), outcomes)


def test_scoring_refuses_a_prompt_with_no_outcomes_at_all() -> None:
    outcomes = [row for row in build_outcomes() if row.trial_id != "sa_00000"]
    with pytest.raises(AnalysisError, match="no outcomes at all"):
        score_conditions(build_forecasts(), outcomes)


def test_the_summary_is_prompt_aggregated() -> None:
    scored = score_conditions(build_forecasts(), build_outcomes())
    entry = scored[TRUE_STATE]
    summary = summarize_condition(entry)
    assert summary.prompt_count == PROMPTS
    assert summary.signed_pair_count == PROMPTS * SIGNED
    assert summary.noop_pair_count == PROMPTS
    assert summary.mae == pytest.approx(prompt_first_mean(entry.prompt_mae()))


def test_the_noop_error_is_reported_separately():
    scored = score_conditions(build_forecasts(), build_outcomes())
    summary = summarize_condition(scored[TRUE_STATE])
    # Every no-op is predicted 0.0 and observed 0.0 in this fixture.
    assert summary.max_abs_noop_error == pytest.approx(0.0)
    assert summary.mean_abs_noop_error == pytest.approx(0.0)


def test_brier_is_omitted_below_the_flip_floor() -> None:
    scored = score_conditions(build_forecasts(), build_outcomes(flip_deltas=3))
    summary = summarize_condition(scored[TRUE_STATE])
    assert summary.observed_flip_count == 3
    assert summary.brier_score is None
    assert summary.brier_min_flips == BRIER_MIN_FLIPS
    assert "below the preregistered floor" in (summary.brier_omitted_reason or "")


def test_brier_is_reported_at_the_flip_floor() -> None:
    scored = score_conditions(build_forecasts(), build_outcomes(flip_deltas=BRIER_MIN_FLIPS))
    summary = summarize_condition(scored[TRUE_STATE])
    assert summary.observed_flip_count == BRIER_MIN_FLIPS
    assert summary.brier_score is not None
    assert summary.brier_omitted_reason is None


def test_a_summary_may_not_report_a_brier_below_the_floor() -> None:
    with pytest.raises(ValueError, match="must be omitted"):
        MethodConditionSummary(
            method_id="m",
            state_condition=StateCondition.TRUE,
            condition_index=0,
            prompt_count=4,
            signed_pair_count=64,
            noop_pair_count=4,
            mae=0.1,
            rmse=0.2,
            sign_accuracy=0.5,
            top_effect_accuracy=0.5,
            interval_coverage=0.9,
            observed_flip_count=2,
            brier_score=0.01,
            brier_min_flips=BRIER_MIN_FLIPS,
            max_abs_noop_error=0.0,
            mean_abs_noop_error=0.0,
        )


# ---------------------------------------------------------------------------
# The decision rule
# ---------------------------------------------------------------------------


def _comparison(true_predict: float, other_predict: float, key) -> PairedComparison:
    scored = score_conditions(
        build_forecasts(
            true_predict=true_predict,
            visible_predict=other_predict,
            wrong_predict=other_predict,
        ),
        build_outcomes(),
    )
    return compare(
        name="test",
        hypothesis="the state helps",
        left=scored[key],
        right=scored[TRUE_STATE],
        direction="positive",
        seed=11,
    )


def test_a_supported_comparison_excludes_zero_in_the_hypothesized_direction() -> None:
    # True state predicts exactly right; the other predicts badly, so the difference is positive.
    comparison = _comparison(true_predict=0.5, other_predict=3.0, key=VISIBLE)
    assert comparison.point_estimate > 0
    assert comparison.excludes_zero is True
    assert comparison.supported is True
    assert "Supported" in comparison.interpretation


def test_a_tie_is_reported_as_no_detected_difference() -> None:
    comparison = _comparison(true_predict=0.5, other_predict=0.5, key=VISIBLE)
    assert comparison.point_estimate == pytest.approx(0.0)
    assert comparison.excludes_zero is False
    assert comparison.supported is False
    assert "No detected difference" in comparison.interpretation
    assert "must not be described as a trend" in comparison.interpretation


def test_a_contradicted_comparison_is_not_reported_as_absence_of_evidence() -> None:
    # The other method predicts exactly right and the true state predicts badly.
    comparison = _comparison(true_predict=3.0, other_predict=0.5, key=VISIBLE)
    assert comparison.point_estimate < 0
    assert comparison.excludes_zero is True
    assert comparison.supported is False
    assert "Contradicted" in comparison.interpretation


def test_the_matched_wrong_state_comparison_uses_the_matched_condition() -> None:
    comparison = _comparison(true_predict=0.5, other_predict=3.0, key=MATCHED_WRONG)
    assert comparison.left == "state_bilinear_ridge:wrong_example:0"
    assert comparison.right == "state_bilinear_ridge:true:0"


def test_a_paired_comparison_cannot_misreport_its_own_decision() -> None:
    with pytest.raises(ValueError, match="supported must follow"):
        PairedComparison(
            name="n",
            hypothesis="h",
            left="a",
            right="b",
            hypothesized_direction="positive",
            point_estimate=0.5,
            ci_low=-0.1,
            ci_high=1.0,
            confidence=0.95,
            resamples=10,
            seed=0,
            prompt_count=4,
            group_count=4,
            excludes_zero=False,
            supported=True,
            interpretation="x",
        )


def test_the_permutations_are_reported_as_a_band() -> None:
    scored = score_conditions(build_forecasts(), build_outcomes())
    band = permutation_band(scored)
    assert band["condition_count"] == 10.0
    assert band["min_mae"] <= band["median_mae"] <= band["max_mae"]


# ---------------------------------------------------------------------------
# Resolution guards
# ---------------------------------------------------------------------------


def _commit_all(run_id: str, records: int = PROMPTS * EXPECTED_RECORDS_PER_PROMPT) -> None:
    for forecast in build_forecasts()[:records]:
        commit_forecast(run_id, forecast)


def test_resolution_refuses_a_dirty_working_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    from causal_self_forecasting.state_audit import resolve as resolve_module

    monkeypatch.setattr(
        resolve_module, "git_state", lambda: {"commit": "abc123", "dirty": True, "branch": "main"}
    )
    with pytest.raises(FinalTestResolutionError, match="uncommitted changes"):
        check_working_tree()


def test_resolution_records_the_commit_from_a_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    from causal_self_forecasting.state_audit import resolve as resolve_module

    monkeypatch.setattr(
        resolve_module, "git_state", lambda: {"commit": "abc123", "dirty": False, "branch": "main"}
    )
    assert check_working_tree() == ("abc123", "main")


def test_resolution_refuses_without_a_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    from causal_self_forecasting.state_audit import resolve as resolve_module

    monkeypatch.setattr(
        resolve_module, "git_state", lambda: {"commit": None, "dirty": False, "branch": None}
    )
    with pytest.raises(FinalTestResolutionError, match="could not read the current git commit"):
        check_working_tree()


def test_resolution_refuses_when_an_outcome_already_exists(isolated_runs: Path) -> None:
    _commit_all(RUN_ID)
    append_jsonl(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS, {"pretend": "outcome"})
    with pytest.raises(FinalTestResolutionError, match="already holds outcome artifacts"):
        check_commitment_state(RUN_ID, PROMPTS)


def test_resolution_refuses_a_wrong_commitment_count(isolated_runs: Path) -> None:
    _commit_all(RUN_ID, records=10)
    with pytest.raises(FinalTestResolutionError, match="preregistered shape"):
        check_commitment_state(RUN_ID, PROMPTS)


def test_resolution_refuses_a_pre_existing_reveal(isolated_runs: Path) -> None:
    _commit_all(RUN_ID)
    reveal_all(RUN_ID)
    with pytest.raises(FinalTestResolutionError, match=r"already holds \d+ reveals"):
        check_commitment_state(RUN_ID, PROMPTS)


def test_resolution_accepts_the_checkpoint_state(isolated_runs: Path) -> None:
    _commit_all(RUN_ID)
    commitments, forecasts, reveals = check_commitment_state(RUN_ID, PROMPTS)
    assert commitments == PROMPTS * EXPECTED_RECORDS_PER_PROMPT
    assert forecasts == commitments
    assert reveals == 0


def test_resolution_refuses_a_missing_salt(isolated_runs: Path) -> None:
    _commit_all(RUN_ID)
    salts = sorted((run_dir(RUN_ID) / "private_payloads" / "salts").glob("*.salt"))
    salts[0].unlink()
    with pytest.raises(FinalTestResolutionError, match="salt files for"):
        check_commitment_state(RUN_ID, PROMPTS)


# ---------------------------------------------------------------------------
# All-candidate reveal, ordering, and target verification
# ---------------------------------------------------------------------------


def test_revealing_every_commitment_without_selection_verifies(isolated_runs: Path) -> None:
    _commit_all(RUN_ID)
    reveals = reveal_all(RUN_ID)
    assert len(reveals) == PROMPTS * EXPECTED_RECORDS_PER_PROMPT
    assert all(reveal.no_selection for reveal in reveals)
    assert all(reveal.selected_intervention_id is None for reveal in reveals)
    assert all(reveal.verified for reveal in reveals)

    report = verify_run_commitments(RUN_ID)
    assert report["verified"] is True
    assert report["checked"] == PROMPTS * EXPECTED_RECORDS_PER_PROMPT
    assert report["failures"] == []
    assert len(read_reveals(RUN_ID)) == PROMPTS * EXPECTED_RECORDS_PER_PROMPT


def test_every_commitment_predates_every_outcome(isolated_runs: Path) -> None:
    _commit_all(RUN_ID)
    write_jsonl(
        run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS,
        [row.model_dump(mode="json") for row in build_outcomes()],
    )
    report = verify_commitment_ordering(RUN_ID)
    assert report["ordering_valid"] is True
    assert report["latest_committed_at"] < report["earliest_observed_at"]
    assert report["final_test_outcomes_exist"] is True


def test_an_outcome_written_before_a_commitment_is_rejected(isolated_runs: Path) -> None:
    """The check that catches resolve-then-commit, which timestamps alone would let through.

    The refusal is unconditional. `allow_existing_seed` exists so a test can construct an
    out-of-order *seed*, and it deliberately does not open a door around the outcome check, since
    that check is the only thing standing between "forecast" and "postdiction".
    """
    from causal_self_forecasting.trials.commitment import ProtocolOrderError

    write_jsonl(
        run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS,
        [row.model_dump(mode="json") for row in build_outcomes()],
    )
    with pytest.raises(ProtocolOrderError, match="is not a forecast"):
        commit_forecast(RUN_ID, build_forecasts()[0])
    with pytest.raises(ProtocolOrderError, match="is not a forecast"):
        commit_forecast(RUN_ID, build_forecasts()[0], allow_existing_seed=True)

    from causal_self_forecasting.paths import FORECAST_COMMITMENTS

    assert not (run_dir(RUN_ID) / FORECAST_COMMITMENTS).exists()


def test_stored_targets_are_independently_verified(isolated_runs: Path) -> None:
    rows = [row.model_dump(mode="json") for row in build_outcomes()]
    write_jsonl(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS, rows)
    assert verify_stored_targets(RUN_ID) == len(rows)


def test_an_edited_target_fails_independent_verification(isolated_runs: Path) -> None:
    rows = [row.model_dump(mode="json") for row in build_outcomes()]
    rows[0]["delta_clean_top_margin"] = 42.0
    write_jsonl(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS, rows)
    with pytest.raises(FinalTestResolutionError, match="does not recompute from its own logits"):
        verify_stored_targets(RUN_ID)


def test_exact_candidate_counts_per_prompt() -> None:
    """17 candidates: 16 signed plus one no-op, and the counts are asserted not assumed."""
    outcomes = build_outcomes()
    by_trial: dict[str, list[StateAuditObservationRecord]] = {}
    for row in outcomes:
        by_trial.setdefault(row.trial_id, []).append(row)
    assert len(by_trial) == PROMPTS
    for rows in by_trial.values():
        assert len(rows) == 17
        assert sum(1 for row in rows if row.is_noop) == 1
        assert sum(1 for row in rows if not row.is_noop) == 16


def test_reveal_refuses_a_commitment_with_no_forecast(isolated_runs: Path) -> None:
    _commit_all(RUN_ID)
    from causal_self_forecasting.paths import FORECASTS

    atomic_write_json(run_dir(RUN_ID) / "unused.json", {})
    (run_dir(RUN_ID) / FORECASTS).unlink()
    with pytest.raises(FinalTestResolutionError, match="has no forecast to reveal"):
        reveal_all(RUN_ID)


def test_reveals_are_written_once_per_commitment(isolated_runs: Path) -> None:
    _commit_all(RUN_ID)
    reveal_all(RUN_ID)
    path = run_dir(RUN_ID) / SELECTION_REVEALS
    assert path.exists()
    assert len(read_reveals(RUN_ID)) == PROMPTS * EXPECTED_RECORDS_PER_PROMPT
