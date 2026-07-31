"""Scoring the sealed forecasts and running the preregistered analysis. Loads no model.

Nothing here fits, refits, tunes, or drops anything. It reads the forecasts that were committed
before the outcomes existed, reads the outcomes, and applies statistics that were fixed in advance.
That is the whole reason it is a separate command from resolution: the analysis can be rerun from
artifacts by anyone, and rerunning it cannot change what was predicted.

The aggregation, from preregistration section 11.1: **average the absolute error over each
prompt's 16 non-no-op interventions first**, giving one number per (method, prompt). Every headline
statistic is a mean over those 32 prompt-level values, never over the 512 pair-level ones, because
pairs inside a prompt share a question and a state and are not independent evidence.

The decision rule, from section 11.3, is applied mechanically and cannot be renegotiated: a
comparison supports its hypothesis only if the 95 percent paired interval excludes zero in the
hypothesized direction. **An interval crossing zero is reported as no detected difference** and is
never described as a trend, a signal, a suggestion, or a direction of travel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..hashing import atomic_write_json, hash_file, read_jsonl, write_jsonl
from ..logging_utils import info
from ..paths import (
    FORECASTS,
    STATE_AUDIT_ANALYSIS,
    STATE_AUDIT_FIGURES,
    STATE_AUDIT_METHOD_SUMMARY,
    STATE_AUDIT_OBSERVATIONS,
    STATE_AUDIT_PAIR_SCORES,
    STATE_AUDIT_PROMPT_SCORES,
    STATE_AUDIT_RESOLUTION,
    run_dir,
)
from ..reproducibility import derive_seed, environment_snapshot, git_state
from ..schemas import (
    FinalTestAnalysisRecord,
    FinalTestResolutionManifest,
    ForecastRecord,
    MethodConditionSummary,
    PairedComparison,
    StateAuditObservationRecord,
    StateCondition,
    compute_final_test_analysis_hash,
)
from ..scoring.metrics import (
    PairScore,
    paired_grouped_bootstrap,
    prompt_first_mean,
    spearman_correlation,
    top_effect_accuracy,
)
from ..trials.commitment import read_forecasts, record_key, verify_run_commitments
from .resolve import load_resolution_manifest

ANALYSIS_ALGORITHM_VERSION = "bluedot_final_test_analysis_v1.0"

# Frozen in preregistration sections 11.3 and 4.4.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED_LABEL = "bluedot.bootstrap"
CONFIDENCE = 0.95
MASTER_SEED = 20260727

# Section 11.4: below this many realized flips the Brier score is omitted, and the omission is
# reported with the realized count. A Brier score over a handful of flips would be quoted.
BRIER_MIN_FLIPS = 20

AGGREGATION = (
    "absolute error averaged over each prompt's 16 non-no-op interventions first, then a mean over "
    "the 32 prompt-level values"
)
DECISION_RULE = (
    "a comparison supports its hypothesis only if the 95 percent paired bootstrap interval over "
    "prompt groups excludes zero in the hypothesized direction; an interval crossing zero is no "
    "detected difference"
)

TRUE_STATE = ("state_bilinear_ridge", StateCondition.TRUE, 0)
VISIBLE = ("visible_information_ridge", StateCondition.NONE, 0)
MATCHED_WRONG = ("state_bilinear_ridge", StateCondition.WRONG_EXAMPLE, 0)


class AnalysisError(RuntimeError):
    """Raised when the analysis cannot be run from the artifacts on disk."""


type ConditionKey = tuple[str, StateCondition, int]


def _label(key: ConditionKey) -> str:
    method, condition, index = key
    return f"{method}:{condition.value}:{index}"


@dataclass(frozen=True)
class ScoredCondition:
    """One method and condition, scored against the outcomes."""

    key: ConditionKey
    pairs: list[PairScore]
    signed_by_prompt: dict[str, list[PairScore]]
    noop_by_prompt: dict[str, list[PairScore]]

    def prompt_mae(self) -> dict[str, float]:
        return {
            prompt: sum(p.absolute_error for p in pairs) / len(pairs)
            for prompt, pairs in self.signed_by_prompt.items()
        }

    def prompt_rmse(self) -> dict[str, float]:
        return {
            prompt: math.sqrt(sum(p.squared_error for p in pairs) / len(pairs))
            for prompt, pairs in self.signed_by_prompt.items()
        }


def read_observations(run_id: str) -> list[StateAuditObservationRecord]:
    return read_observations_at(run_dir(run_id))


def read_observations_at(directory: Path) -> list[StateAuditObservationRecord]:
    """Read outcomes from any directory holding them, a run directory or a public bundle."""
    path = directory / STATE_AUDIT_OBSERVATIONS
    if not path.exists():
        raise AnalysisError(f"no outcomes at {path}; the final test has not been resolved")
    return [StateAuditObservationRecord.model_validate(row) for row in read_jsonl(path)]


def read_forecasts_at(directory: Path) -> list[ForecastRecord]:
    """Read sealed forecasts from any directory holding them."""
    path = directory / FORECASTS
    if not path.exists():
        raise AnalysisError(f"no forecasts at {path}")
    return [ForecastRecord.model_validate(row) for row in read_jsonl(path)]


def score_conditions(
    forecasts: Sequence[ForecastRecord],
    observations: Sequence[StateAuditObservationRecord],
) -> dict[ConditionKey, ScoredCondition]:
    """Match every sealed forecast to its outcome. No refitting, no dropping.

    A forecast naming a candidate with no outcome, or an outcome with no forecast, is an error
    rather than something to skip: silently scoring a subset would report a number over whichever
    pairs happened to survive.
    """
    outcome_by_pair = {(o.trial_id, o.candidate_id): o for o in observations}
    group_by_trial = {o.trial_id: o.group_id for o in observations}

    scored: dict[ConditionKey, ScoredCondition] = {}
    for forecast in forecasts:
        key: ConditionKey = (
            forecast.method_id,
            forecast.state_condition,
            forecast.condition_index,
        )
        entry = scored.get(key)
        if entry is None:
            entry = ScoredCondition(key=key, pairs=[], signed_by_prompt={}, noop_by_prompt={})
            scored[key] = entry

        group_id = group_by_trial.get(forecast.trial_id)
        if group_id is None:
            raise AnalysisError(
                f"forecast for {forecast.trial_id} under {_label(key)} has no outcomes at all"
            )

        for candidate in forecast.candidate_forecasts:
            outcome = outcome_by_pair.get((forecast.trial_id, candidate.intervention_id))
            if outcome is None:
                raise AnalysisError(
                    f"{forecast.trial_id}/{candidate.intervention_id} was forecast under "
                    f"{_label(key)} but has no outcome"
                )
            pair = PairScore(
                trial_id=forecast.trial_id,
                method_id=_label(key),
                intervention_id=candidate.intervention_id,
                group_id=group_id,
                split="final_test",
                mechanism="noop" if outcome.is_noop else "residual_add",
                is_noop=outcome.is_noop,
                predicted_delta=candidate.delta_margin_mean,
                observed_delta=outcome.delta_clean_top_margin,
                predicted_flip_probability=candidate.p_answer_flip,
                observed_flip=outcome.answer_flip,
                interval_low=candidate.delta_margin_q05,
                interval_high=candidate.delta_margin_q95,
            )
            entry.pairs.append(pair)
            bucket = entry.noop_by_prompt if outcome.is_noop else entry.signed_by_prompt
            bucket.setdefault(group_id, []).append(pair)

    if not scored:
        raise AnalysisError("no forecasts to score")
    return scored


def summarize_condition(entry: ScoredCondition) -> MethodConditionSummary:
    """Every reported metric for one condition, prompt-aggregated where that applies."""
    method, condition, index = entry.key
    signed = [pair for pair in entry.pairs if not pair.is_noop]
    noops = [pair for pair in entry.pairs if pair.is_noop]
    if not signed:
        raise AnalysisError(f"{_label(entry.key)} has no non-no-op pairs to score")

    flips = sum(1 for pair in signed if pair.observed_flip)
    brier = None
    reason = None
    if flips >= BRIER_MIN_FLIPS:
        brier = sum(pair.flip_brier for pair in signed) / len(signed)
    else:
        reason = (
            f"only {flips} realized answer flips across the final test, below the preregistered "
            f"floor of {BRIER_MIN_FLIPS}; a Brier score over this few flips would be quoted as if "
            "it meant something"
        )

    # Spearman over the pair level: the metric is about ranking effects, and ranking within a
    # prompt is what the top-effect metric covers separately.
    spearman = spearman_correlation(
        [pair.predicted_delta for pair in signed], [pair.observed_delta for pair in signed]
    )
    trials = [pairs for pairs in entry.signed_by_prompt.values() if len(pairs) >= 2]

    return MethodConditionSummary(
        method_id=method,
        state_condition=condition,
        condition_index=index,
        prompt_count=len(entry.signed_by_prompt),
        signed_pair_count=len(signed),
        noop_pair_count=len(noops),
        mae=prompt_first_mean(entry.prompt_mae()),
        rmse=prompt_first_mean(entry.prompt_rmse()),
        sign_accuracy=sum(int(pair.sign_correct) for pair in signed) / len(signed),
        spearman=spearman,
        top_effect_accuracy=top_effect_accuracy(trials) if trials else 0.0,
        interval_coverage=sum(int(pair.interval_covered) for pair in signed) / len(signed),
        observed_flip_count=flips,
        brier_score=brier,
        brier_min_flips=BRIER_MIN_FLIPS,
        brier_omitted_reason=reason,
        max_abs_noop_error=max((pair.absolute_error for pair in noops), default=0.0),
        mean_abs_noop_error=(
            sum(pair.absolute_error for pair in noops) / len(noops) if noops else 0.0
        ),
    )


def _interpret(name: str, difference, hypothesis: str, direction: str) -> tuple[bool, str]:
    supported = difference.excludes_zero and (
        difference.point_estimate > 0 if direction == "positive" else difference.point_estimate < 0
    )
    if not difference.excludes_zero:
        return False, (
            f"No detected difference. The 95 percent paired interval "
            f"[{difference.ci_low:.6f}, {difference.ci_high:.6f}] crosses zero, so {hypothesis} is "
            "not supported. Under the preregistered decision rule this is a null result and must "
            "not be described as a trend or a direction of travel."
        )
    if supported:
        return True, (
            f"Supported. The interval [{difference.ci_low:.6f}, {difference.ci_high:.6f}] excludes "
            f"zero in the hypothesized direction, so {hypothesis} is supported at this sample size."
        )
    return False, (
        f"Contradicted. The interval [{difference.ci_low:.6f}, {difference.ci_high:.6f}] excludes "
        f"zero in the direction opposite to {hypothesis}, which counts against it rather than "
        "being an absence of evidence."
    )


def compare(
    name: str,
    hypothesis: str,
    left: ScoredCondition,
    right: ScoredCondition,
    direction: str,
    seed: int,
) -> PairedComparison:
    """One preregistered paired difference, with the decision rule applied mechanically."""
    left_mae = left.prompt_mae()
    right_mae = right.prompt_mae()
    difference = paired_grouped_bootstrap(
        left_mae, right_mae, resamples=BOOTSTRAP_RESAMPLES, confidence=CONFIDENCE, seed=seed
    )
    supported, interpretation = _interpret(name, difference, hypothesis, direction)
    return PairedComparison(
        name=name,
        hypothesis=hypothesis,
        left=_label(left.key),
        right=_label(right.key),
        hypothesized_direction="positive" if direction == "positive" else "negative",
        point_estimate=difference.point_estimate,
        ci_low=difference.ci_low,
        ci_high=difference.ci_high,
        confidence=CONFIDENCE,
        resamples=difference.resamples,
        seed=seed,
        prompt_count=len(left_mae),
        group_count=difference.group_count,
        excludes_zero=difference.excludes_zero,
        supported=supported,
        interpretation=interpretation,
    )


def permutation_band(scored: dict[ConditionKey, ScoredCondition]) -> dict[str, float]:
    """The spread of the ten shuffled-state controls, as a band, not ten tests.

    Section 9 is explicit that the permutations are reported as a robustness band around the
    primary matched control rather than as ten separate comparisons, because ten tests invite
    picking the convenient one.
    """
    values = [
        prompt_first_mean(entry.prompt_mae())
        for key, entry in scored.items()
        if key[1] is StateCondition.SHUFFLED
    ]
    if not values:
        return {}
    array = np.asarray(sorted(values), dtype=np.float64)
    return {
        "condition_count": float(len(values)),
        "min_mae": float(array.min()),
        "median_mae": float(np.median(array)),
        "max_mae": float(array.max()),
        "mean_mae": float(array.mean()),
    }


def _write_tables(run_id: str, scored: dict[ConditionKey, ScoredCondition]) -> dict[str, str]:
    """Machine-readable tables: one row per scored pair, and one per (condition, prompt)."""
    directory = run_dir(run_id)
    pair_rows = [
        {
            "method_condition": pair.method_id,
            "trial_id": pair.trial_id,
            "group_id": pair.group_id,
            "candidate_id": pair.intervention_id,
            "is_noop": pair.is_noop,
            "predicted_delta": pair.predicted_delta,
            "observed_delta": pair.observed_delta,
            "absolute_error": pair.absolute_error,
            "squared_error": pair.squared_error,
            "sign_correct": pair.sign_correct,
            "interval_covered": pair.interval_covered,
            "predicted_flip_probability": pair.predicted_flip_probability,
            "observed_flip": pair.observed_flip,
        }
        for entry in scored.values()
        for pair in entry.pairs
    ]
    write_jsonl(directory / STATE_AUDIT_PAIR_SCORES, pair_rows)

    prompt_rows = []
    for key, entry in sorted(scored.items(), key=lambda item: _label(item[0])):
        mae = entry.prompt_mae()
        rmse = entry.prompt_rmse()
        for prompt in sorted(mae):
            signed = entry.signed_by_prompt[prompt]
            prompt_rows.append(
                {
                    "method_condition": _label(key),
                    "method_id": key[0],
                    "state_condition": key[1].value,
                    "condition_index": key[2],
                    "group_id": prompt,
                    "signed_pairs": len(signed),
                    "mae": mae[prompt],
                    "rmse": rmse[prompt],
                    "sign_accuracy": sum(int(p.sign_correct) for p in signed) / len(signed),
                    "observed_flips": sum(1 for p in signed if p.observed_flip),
                }
            )
    write_jsonl(directory / STATE_AUDIT_PROMPT_SCORES, prompt_rows)
    return {
        "pair_scores": hash_file(directory / STATE_AUDIT_PAIR_SCORES),
        "prompt_scores": hash_file(directory / STATE_AUDIT_PROMPT_SCORES),
    }


def _write_figures(
    run_id: str,
    summaries: Sequence[MethodConditionSummary],
    comparisons: Sequence[PairedComparison],
    scored: dict[ConditionKey, ScoredCondition],
) -> list[str]:
    """Two minimal figures, drawn from the computed numbers only.

    No hard-coded values anywhere: every coordinate comes from the summaries and comparisons this
    run produced. Matplotlib's non-interactive backend is selected explicitly so this works on a
    headless machine.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = run_dir(run_id) / STATE_AUDIT_FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1. Prompt-aggregated MAE per method and condition, permutations collapsed to a band.
    primary = [
        summary for summary in summaries if summary.state_condition is not StateCondition.SHUFFLED
    ]
    labels = [f"{s.method_id}\n{s.state_condition.value}" for s in primary]
    values = [s.mae for s in primary]
    band = permutation_band(scored)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = range(len(values))
    axis.barh(list(positions), values, color="#4a6f8a")
    if band:
        axis.axvspan(
            band["min_mae"],
            band["max_mae"],
            color="#c9a227",
            alpha=0.25,
            label=f"shuffled-state band (n={int(band['condition_count'])})",
        )
        axis.legend(loc="lower right", fontsize=8)
    axis.set_yticks(list(positions))
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("prompt-aggregated mean absolute error on delta_clean_top_margin")
    axis.set_title(
        f"Final-test MAE by method and state condition ({primary[0].prompt_count} prompts)",
        fontsize=10,
    )
    axis.invert_yaxis()
    figure.tight_layout()
    path = directory / "final_test_mae_by_method.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(str(path.name))

    # 2. The two primary paired differences with their bootstrap intervals.
    figure, axis = plt.subplots(figsize=(8, 3.2))
    for offset, comparison in enumerate(comparisons):
        axis.errorbar(
            comparison.point_estimate,
            offset,
            xerr=[
                [comparison.point_estimate - comparison.ci_low],
                [comparison.ci_high - comparison.point_estimate],
            ],
            fmt="o",
            color="#2f5d50" if comparison.supported else "#8a4a4a",
            capsize=4,
        )
    axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(range(len(comparisons)))
    axis.set_yticklabels([c.name for c in comparisons], fontsize=8)
    axis.set_xlabel("paired difference in prompt-aggregated MAE (95 percent bootstrap interval)")
    axis.set_title(
        f"Primary comparisons, {BOOTSTRAP_RESAMPLES} paired resamples over prompt groups",
        fontsize=10,
    )
    axis.invert_yaxis()
    figure.tight_layout()
    path = directory / "final_test_primary_comparisons.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(str(path.name))
    return written


def analyze_final_test(run_id: str, training_run_id: str, force: bool = False) -> dict[str, Any]:
    """Score the sealed forecasts and run the preregistered analysis. Loads no model."""
    directory = run_dir(run_id)
    analysis_path = directory / STATE_AUDIT_ANALYSIS
    if analysis_path.exists() and not force:
        raise AnalysisError(
            f"{analysis_path} already exists. The analysis is run once; rerunning it after seeing "
            "the numbers is how a decision rule gets renegotiated. Pass force only to repair a "
            "documented bug, and record the bug and the change in docs/experiment_log.md."
        )

    resolution = load_resolution_manifest(run_id)
    if resolution.status != "complete":
        raise AnalysisError(
            f"the resolution of run {run_id!r} is marked {resolution.status!r}; a run that did not "
            "finish is analyzed as a run that did not finish, not as a result"
        )

    verification = verify_run_commitments(run_id)
    forecasts = read_forecasts(run_id)
    observations = read_observations(run_id)

    keys = [record_key(forecast) for forecast in forecasts]
    if len(set(keys)) != len(keys):
        raise AnalysisError("the forecasts repeat a key; the set cannot be scored unambiguously")

    scored = score_conditions(forecasts, observations)
    summaries = [
        summarize_condition(entry)
        for _, entry in sorted(scored.items(), key=lambda i: _label(i[0]))
    ]
    by_key = {entry.key: entry for entry in scored.values()}

    for key in (TRUE_STATE, VISIBLE, MATCHED_WRONG):
        if key not in by_key:
            raise AnalysisError(f"the primary comparison needs {_label(key)}, which was not scored")

    seed = derive_seed(BOOTSTRAP_SEED_LABEL, MASTER_SEED)
    comparisons = [
        compare(
            name="visible_information minus true_state",
            hypothesis=(
                "H-BD1, that the state-conditioned ridge predicts the target with lower "
                "prompt-aggregated MAE than the visible-information ridge"
            ),
            left=by_key[VISIBLE],
            right=by_key[TRUE_STATE],
            direction="positive",
            seed=seed,
        ),
        compare(
            name="matched_wrong_state minus true_state",
            hypothesis=(
                "H-BD2, that the advantage depends on the state belonging to the prompt being "
                "predicted, so substituting a matched wrong state raises MAE"
            ),
            left=by_key[MATCHED_WRONG],
            right=by_key[TRUE_STATE],
            direction="positive",
            seed=seed,
        ),
    ]

    table_hashes = _write_tables(run_id, scored)
    figures = _write_figures(run_id, summaries, comparisons, scored)

    summary_payload = {
        "algorithm_version": ANALYSIS_ALGORITHM_VERSION,
        "run_id": run_id,
        "aggregation": AGGREGATION,
        "prompt_count": summaries[0].prompt_count,
        "methods": [summary.model_dump(mode="json") for summary in summaries],
        "permutation_band": permutation_band(scored),
        "table_hashes": table_hashes,
    }
    atomic_write_json(directory / STATE_AUDIT_METHOD_SUMMARY, summary_payload)

    payload: dict[str, Any] = {
        "schema_version": FinalTestAnalysisRecord.model_fields["schema_version"].default,
        "analysis_id": f"{run_id}_analysis_v1",
        "study_id": resolution.study_id,
        "final_test_run_id": run_id,
        "training_run_id": training_run_id,
        "target_name": "delta_clean_top_margin",
        "layer": resolution.layer,
        "norm_ratio": resolution.norm_ratio,
        "global_alpha": resolution.global_alpha,
        "prompt_count": summaries[0].prompt_count,
        "group_count": summaries[0].prompt_count,
        "aggregation": AGGREGATION,
        "decision_rule": DECISION_RULE,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
        "method_summaries": [summary.model_dump(mode="json") for summary in summaries],
        "primary_comparisons": [comparison.model_dump(mode="json") for comparison in comparisons],
        "permutation_band": permutation_band(scored),
        "resolution_manifest_hash": resolution.manifest_hash,
        "forecasts_hash": hash_file(directory / FORECASTS),
        "observations_hash": resolution.observations_hash,
        "commitments_verified": bool(verification["verified"]),
        "scientific_forecast_evaluation": bool(verification["verified"])
        and resolution.status == "complete",
    }

    record = FinalTestAnalysisRecord(
        **payload,
        analysis_hash=compute_final_test_analysis_hash(payload),
        environment=environment_snapshot(),
        code_commit=(git_state().get("commit")),
        notes=(
            "Analysis run once from sealed forecasts and verified outcomes. Nothing was fitted, "
            "refitted, tuned, or dropped here. An interval crossing zero is no detected difference."
        ),
    )
    atomic_write_json(analysis_path, record.model_dump(mode="json"))

    info(
        "analyzed the final test",
        run_id=run_id,
        conditions=len(summaries),
        prompts=summaries[0].prompt_count,
        verified=verification["verified"],
    )
    return analysis_report(record, figures, table_hashes, directory)


def analyze_report_paths(run_id: str) -> dict[str, Path]:
    directory = run_dir(run_id)
    return {
        "analysis": directory / STATE_AUDIT_ANALYSIS,
        "method_summary": directory / STATE_AUDIT_METHOD_SUMMARY,
        "pair_scores": directory / STATE_AUDIT_PAIR_SCORES,
        "prompt_scores": directory / STATE_AUDIT_PROMPT_SCORES,
        "figures": directory / STATE_AUDIT_FIGURES,
    }


def analysis_report(
    record: FinalTestAnalysisRecord,
    figures: Sequence[str],
    table_hashes: dict[str, str],
    directory: Path,
) -> dict[str, Any]:
    headline = {
        summary.method_id + ":" + summary.state_condition.value: summary.mae
        for summary in record.method_summaries
        if summary.state_condition is not StateCondition.SHUFFLED
    }
    return {
        "analysis_id": record.analysis_id,
        "final_test_run_id": record.final_test_run_id,
        "training_run_id": record.training_run_id,
        "analysis_hash": record.analysis_hash,
        "algorithm_version": ANALYSIS_ALGORITHM_VERSION,
        "scientific_forecast_evaluation": record.scientific_forecast_evaluation,
        "commitments_verified": record.commitments_verified,
        "target_name": record.target_name,
        "layer": record.layer,
        "norm_ratio": record.norm_ratio,
        "global_alpha": record.global_alpha,
        "prompt_count": record.prompt_count,
        "group_count": record.group_count,
        "aggregation": record.aggregation,
        "decision_rule": record.decision_rule,
        "bootstrap": {"resamples": record.bootstrap_resamples, "seed": record.bootstrap_seed},
        "headline_mae": headline,
        "permutation_band": record.permutation_band,
        "primary_comparisons": [
            {
                "name": comparison.name,
                "left": comparison.left,
                "right": comparison.right,
                "point_estimate": comparison.point_estimate,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "prompt_count": comparison.prompt_count,
                "group_count": comparison.group_count,
                "excludes_zero": comparison.excludes_zero,
                "supported": comparison.supported,
                "interpretation": comparison.interpretation,
            }
            for comparison in record.primary_comparisons
        ],
        "method_summaries": [
            summary.model_dump(mode="json") for summary in record.method_summaries
        ],
        "artifacts": {
            "analysis": STATE_AUDIT_ANALYSIS,
            "method_summary": STATE_AUDIT_METHOD_SUMMARY,
            "pair_scores": STATE_AUDIT_PAIR_SCORES,
            "prompt_scores": STATE_AUDIT_PROMPT_SCORES,
            "figures": [f"{STATE_AUDIT_FIGURES}/{name}" for name in figures],
        },
        "table_hashes": table_hashes,
        "run_path": str(directory),
        "notes": record.notes,
    }


def replay_analysis(run_id: str) -> dict[str, Any]:
    """Recompute the analysis from a run directory and compare it to the stored record."""
    return replay_from_directory(run_dir(run_id), source=run_id)


def replay_from_directory(directory: Path, *, source: str) -> dict[str, Any]:
    """Recompute the analysis from any directory of artifacts. Loads no model.

    The model-free replay a third party runs. It recomputes every summary and both comparisons from
    the forecasts and outcomes on disk and checks them against what was written, so a stored
    analysis that does not follow from its own inputs is detectable without rerunning the model.

    Directory-based rather than run-id-based so that the published bundle replays through exactly
    this function. A separate reimplementation for the public bundle could agree with the stored
    analysis while disagreeing with the code that produced it, which is the failure a replay exists
    to catch.
    """
    analysis_path = directory / STATE_AUDIT_ANALYSIS
    if not analysis_path.exists():
        raise AnalysisError(f"no analysis to replay at {analysis_path}")
    from ..hashing import read_json

    stored = FinalTestAnalysisRecord.model_validate(read_json(analysis_path))
    resolution_path = directory / STATE_AUDIT_RESOLUTION
    if not resolution_path.exists():
        raise AnalysisError(f"no final-test resolution at {resolution_path}")
    resolution = FinalTestResolutionManifest.model_validate(read_json(resolution_path))
    failures: list[str] = []

    # Hash the files on disk, not the values two records happen to agree on. Comparing the
    # resolution's stored hash against the analysis's stored hash would pass even after the
    # outcomes file underneath both of them had been rewritten.
    if hash_file(directory / FORECASTS) != stored.forecasts_hash:
        failures.append("the forecasts on disk do not match the hash the analysis cites")

    observations_path = directory / STATE_AUDIT_OBSERVATIONS
    if not observations_path.exists():
        failures.append("the outcomes the analysis cites are missing")
    else:
        on_disk = hash_file(observations_path)
        if on_disk != stored.observations_hash:
            failures.append(
                f"the outcomes on disk hash to {on_disk} but the analysis cites "
                f"{stored.observations_hash}"
            )
        if on_disk != resolution.observations_hash:
            failures.append(
                "the outcomes on disk do not match the hash the resolution manifest recorded"
            )

    if resolution.manifest_hash != stored.resolution_manifest_hash:
        failures.append("the resolution manifest does not match the hash the analysis cites")

    scored = score_conditions(read_forecasts_at(directory), read_observations_at(directory))
    recomputed = {
        (s.method_id, s.state_condition, s.condition_index): s
        for s in (summarize_condition(entry) for entry in scored.values())
    }
    for summary in stored.method_summaries:
        key = (summary.method_id, summary.state_condition, summary.condition_index)
        fresh = recomputed.get(key)
        if fresh is None:
            failures.append(f"{_label(key)} is in the stored analysis but not recomputable")
            continue
        for field in ("mae", "rmse", "sign_accuracy", "interval_coverage"):
            if abs(getattr(fresh, field) - getattr(summary, field)) > 1e-9:
                failures.append(f"{_label(key)}: {field} does not recompute")
        if fresh.observed_flip_count != summary.observed_flip_count:
            failures.append(f"{_label(key)}: flip count does not recompute")

    by_key = {entry.key: entry for entry in scored.values()}
    plans = [
        ("visible_information minus true_state", VISIBLE, TRUE_STATE),
        ("matched_wrong_state minus true_state", MATCHED_WRONG, TRUE_STATE),
    ]
    for (name, left, right), comparison in zip(plans, stored.primary_comparisons, strict=False):
        fresh = compare(
            name=name,
            hypothesis=comparison.hypothesis,
            left=by_key[left],
            right=by_key[right],
            direction=comparison.hypothesized_direction,
            seed=stored.bootstrap_seed,
        )
        for field in ("point_estimate", "ci_low", "ci_high"):
            if abs(getattr(fresh, field) - getattr(comparison, field)) > 1e-9:
                failures.append(f"{name}: {field} does not recompute")
        if fresh.supported != comparison.supported:
            failures.append(f"{name}: the decision does not recompute")

    return {
        "run_id": source,
        "source_directory": str(directory),
        "analysis_id": stored.analysis_id,
        "analysis_hash": stored.analysis_hash,
        "conditions_checked": len(stored.method_summaries),
        "comparisons_checked": len(stored.primary_comparisons),
        "bootstrap_seed": stored.bootstrap_seed,
        "bootstrap_resamples": stored.bootstrap_resamples,
        "valid": not failures,
        "failures": failures,
        "notes": (
            "Model-free replay. Every summary and both primary comparisons were recomputed from "
            "the forecasts and outcomes on disk and compared to the stored analysis."
        ),
    }


__all__ = [
    "AGGREGATION",
    "ANALYSIS_ALGORITHM_VERSION",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED_LABEL",
    "BRIER_MIN_FLIPS",
    "DECISION_RULE",
    "MATCHED_WRONG",
    "TRUE_STATE",
    "VISIBLE",
    "AnalysisError",
    "ScoredCondition",
    "analyze_final_test",
    "analyze_report_paths",
    "compare",
    "permutation_band",
    "read_forecasts_at",
    "read_observations",
    "read_observations_at",
    "replay_analysis",
    "replay_from_directory",
    "score_conditions",
    "summarize_condition",
]
