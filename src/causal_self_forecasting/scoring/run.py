"""Score a resolved run: match committed forecasts to observations and aggregate.

The scorer never invents a result. It reads forecasts committed before the interventions were
selected and observations measured after, matches them by trial and candidate, and reports
metrics with their sample counts and exclusions. It writes into the run directory only, never
to the public results tree, and it refuses to call a run scientific unless its commitments
verified.
"""

from __future__ import annotations

from typing import Any

from ..hashing import atomic_write_json, read_json, write_jsonl
from ..logging_utils import info
from ..paths import (
    RESOLUTION_MANIFEST,
    SCORE_RECORDS,
    SCORES,
    run_dir,
)
from ..schemas import Mechanism, ScoreRecord, StateCondition
from ..trials.commitment import read_forecasts
from ..trials.generate import TrialGenerationError, read_trials
from ..trials.resolve import read_observations
from .metrics import (
    PairScore,
    brier_score,
    grouped_bootstrap_ci,
    interval_coverage,
    log_loss,
    mae,
    rmse,
    sign_accuracy,
    top_effect_accuracy,
)


class ScoringError(RuntimeError):
    """Raised when a run cannot be scored as requested."""


def _resolution_manifest(run_id: str) -> dict[str, Any]:
    path = run_dir(run_id) / RESOLUTION_MANIFEST
    if not path.exists():
        raise ScoringError(
            f"run {run_id!r} has not been resolved; run `csf trials resolve --run-id {run_id}` "
            "first"
        )
    return read_json(path)


def _metric_block(scores: list[PairScore]) -> dict[str, Any]:
    """Numeric and flip metrics over a set of pairs, with grouped bootstrap intervals."""
    if not scores:
        return {"n": 0, "n_groups": 0}
    group_ids = sorted({score.group_id for score in scores})
    mae_low, mae_high = grouped_bootstrap_ci(scores, mae)
    brier_low, brier_high = grouped_bootstrap_ci(scores, brier_score)
    return {
        "n": len(scores),
        "n_groups": len(group_ids),
        "mae": mae(scores),
        "mae_ci": [mae_low, mae_high],
        "rmse": rmse(scores),
        "sign_accuracy": sign_accuracy(scores),
        "interval_coverage": interval_coverage(scores),
        "flip_brier": brier_score(scores),
        "flip_brier_ci": [brier_low, brier_high],
        "flip_log_loss": log_loss(scores),
    }


def _score_record(score: PairScore, state_condition: StateCondition) -> ScoreRecord:
    return ScoreRecord(
        trial_id=score.trial_id,
        method_id=score.method_id,
        intervention_id=score.intervention_id,
        state_condition=state_condition,
        split=score.split,  # type: ignore[arg-type]
        mechanism=Mechanism(score.mechanism),
        group_id=score.group_id,
        absolute_error=score.absolute_error,
        squared_error=score.squared_error,
        sign_correct=score.sign_correct,
        interval_covered=score.interval_covered,
        interval_width=score.interval_width,
        flip_brier=score.flip_brier,
        flip_log_loss=score.flip_log_loss,
        predicted_delta=score.predicted_delta,
        observed_delta=score.observed_delta,
        predicted_flip_probability=score.predicted_flip_probability,
        observed_flip=score.observed_flip,
    )


def score_run(run_id: str, force: bool = False) -> dict[str, Any]:
    """Score every committed method in a resolved run."""
    directory = run_dir(run_id)
    if not directory.exists():
        raise ScoringError(f"no such run: {directory}")
    if (directory / SCORES).exists() and not force:
        raise ScoringError(f"{directory / SCORES} already exists; pass force to rescore")

    resolution = _resolution_manifest(run_id)
    classification = resolution.get("classification", "unknown")
    fixture_only = bool(resolution.get("fixture_only", False))
    is_scientific = classification == "scientific_forecast_resolution"
    commitments_verified = bool(resolution.get("commitments_verified", False))

    if is_scientific and not commitments_verified:
        raise ScoringError(
            f"run {run_id!r} is a scientific forecast resolution whose commitments did not "
            "verify; it must not be scored as a scientific result"
        )

    try:
        trials = read_trials(run_id)
    except TrialGenerationError as error:
        raise ScoringError(str(error)) from error
    trial_by_id = {trial.trial_id: trial for trial in trials}

    forecasts = read_forecasts(run_id)
    if not forecasts:
        raise ScoringError(f"run {run_id!r} has no committed forecasts to score")

    observations = {(obs.trial_id, obs.intervention_id): obs for obs in read_observations(run_id)}
    if not observations:
        raise ScoringError(f"run {run_id!r} has no observations; resolve it before scoring")

    methods: dict[str, list[PairScore]] = {}
    method_state_condition: dict[str, StateCondition] = {}
    trials_by_method: dict[str, dict[str, list[PairScore]]] = {}

    forecasts_without_observation = 0
    matched_pairs = 0

    for forecast in forecasts:
        trial = trial_by_id.get(forecast.trial_id)
        if trial is None:
            continue
        method_state_condition[forecast.method_id] = forecast.state_condition
        for candidate in forecast.candidate_forecasts:
            observation = observations.get((forecast.trial_id, candidate.intervention_id))
            if observation is None:
                forecasts_without_observation += 1
                continue
            score = PairScore(
                trial_id=forecast.trial_id,
                method_id=forecast.method_id,
                intervention_id=candidate.intervention_id,
                group_id=trial.group_id,
                split=trial.split.value,
                mechanism=observation.mechanism.value,
                is_noop=observation.mechanism is Mechanism.NOOP,
                predicted_delta=candidate.delta_margin_mean,
                observed_delta=observation.delta_margin,
                predicted_flip_probability=candidate.p_answer_flip,
                observed_flip=observation.answer_flip,
                interval_low=candidate.delta_margin_q05,
                interval_high=candidate.delta_margin_q95,
            )
            methods.setdefault(forecast.method_id, []).append(score)
            trials_by_method.setdefault(forecast.method_id, {}).setdefault(
                forecast.trial_id, []
            ).append(score)
            matched_pairs += 1

    if matched_pairs == 0:
        raise ScoringError(
            f"run {run_id!r} has forecasts and observations but none matched by trial and "
            "candidate; the forecasts and the resolution do not refer to the same interventions"
        )

    all_records: list[ScoreRecord] = []
    method_reports: dict[str, Any] = {}
    for method_id, scores in sorted(methods.items()):
        headline = [score for score in scores if not score.is_noop]
        state_condition = method_state_condition[method_id]

        # Ranking needs every candidate in a trial observed. A forecast-mode run observes only
        # the selected candidate, so those trials do not contribute and the count says so.
        complete_trials = [
            trial_scores
            for trial_id, trial_scores in trials_by_method[method_id].items()
            if len(trial_scores) == _forecast_candidate_count(forecasts, method_id, trial_id)
            and len(trial_scores) >= 2
        ]
        ranking: dict[str, Any] = {"n_trials": len(complete_trials)}
        if complete_trials:
            ranking["top_effect_accuracy"] = top_effect_accuracy(complete_trials)

        method_reports[method_id] = {
            "state_condition": state_condition.value,
            "headline_excludes_noop": _metric_block(headline),
            "all_candidates_including_noop": _metric_block(scores),
            "candidate_ranking": ranking,
            "counts": {
                "matched_pairs": len(scores),
                "noop_pairs": sum(int(score.is_noop) for score in scores),
                "headline_pairs": len(headline),
            },
        }
        all_records.extend(_score_record(score, state_condition) for score in scores)

    write_jsonl(directory / SCORE_RECORDS, all_records)

    report = {
        "schema_version": "1.0",
        "run_id": run_id,
        "phase": "score",
        "classification": classification,
        "scientific_result": False,
        "scored_as_scientific": is_scientific,
        "fixture_only": fixture_only,
        "commitments_verified": commitments_verified,
        "methods": method_reports,
        "exclusions": {
            "forecasts_without_observation": forecasts_without_observation,
            "resolution_failures": int(resolution.get("counts", {}).get("failures", 0)),
            "note": (
                "No-op candidates are excluded from headline metrics and reported under "
                "all_candidates_including_noop. Ranking uses only trials where every candidate "
                "was observed."
            ),
        },
        "notes": (
            "Scores are written to the run directory only. This is not a public result and the "
            "exporter does not read it."
        ),
    }
    atomic_write_json(directory / SCORES, report)

    info(
        "scored run",
        run_id=run_id,
        methods=len(method_reports),
        matched_pairs=matched_pairs,
        scientific=is_scientific,
    )
    return report


def _forecast_candidate_count(forecasts: list[Any], method_id: str, trial_id: str) -> int:
    for forecast in forecasts:
        if forecast.method_id == method_id and forecast.trial_id == trial_id:
            return len(forecast.candidate_forecasts)
    return 0


__all__ = ["ScoringError", "score_run"]
