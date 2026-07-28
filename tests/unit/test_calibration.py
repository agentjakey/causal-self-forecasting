"""Tests for global intervention strength, the pass conditions, and the layer state machine.

All offline, all pure. Nothing here loads a model, and the fixtures are hand-built so that the
boundary cases land exactly on the thresholds rather than near them.

The failure these guard against is a calibration that quietly chooses a strength on the wrong
grounds: a per-prompt alpha that leaks the state norm into the published strength, a boundary
comparison that silently moves the grid, or a fallback taken after the primary layer already
succeeded.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from causal_self_forecasting.calibration.criteria import EffectSample, summarize_ratio
from causal_self_forecasting.calibration.plan import (
    CalibrationPlanError,
    build_calibration_plan,
    build_decision_record,
    build_forward_counts,
    load_calibration_plan,
    plan_content_bytes,
    verify_calibration_plan,
    write_calibration_plan,
)
from causal_self_forecasting.calibration.selection import (
    SelectionError,
    select_calibration_ratio,
)
from causal_self_forecasting.calibration.strength import (
    NORM_RATIOS,
    StrengthError,
    alpha_for_ratio,
    alpha_table,
    check_global_alpha,
    reference_norm,
    validate_calibration_norms,
)
from causal_self_forecasting.config import CalibrationPlanConfig, load_config
from causal_self_forecasting.schemas import (
    CalibrationPlanRecord,
    CalibrationStatus,
    CalibrationThresholds,
    compute_calibration_decision_hash,
    compute_calibration_plan_hash,
)

THRESHOLDS = CalibrationThresholds(
    min_large_effect_fraction=0.15,
    large_effect_threshold=0.10,
    min_median_abs_effect=0.05,
    max_p95_abs_effect=4.0,
)
NOOP_TOLERANCE = 1.0e-3

PROMPTS = [f"g{index:04d}" for index in range(32)]


# ---------------------------------------------------------------------------
# Global strength
# ---------------------------------------------------------------------------


def _norms(values: list[float]) -> dict[str, float]:
    return dict(zip(PROMPTS, values, strict=True))


def test_reference_norm_is_the_median_of_the_supplied_norms() -> None:
    values = [float(index + 1) for index in range(32)]
    assert reference_norm(_norms(values), PROMPTS) == pytest.approx(16.5)


def test_even_sample_median_averages_the_two_central_values() -> None:
    """NumPy's convention, recorded in the plan, pinned here."""
    prompts = ["a", "b", "c", "d"]
    norms = {"a": 1.0, "b": 2.0, "c": 4.0, "d": 8.0}
    assert reference_norm(norms, prompts) == pytest.approx(3.0)
    assert reference_norm(norms, prompts) == pytest.approx(float(np.median([1.0, 2.0, 4.0, 8.0])))


def test_alignment_is_by_prompt_identity_not_input_order() -> None:
    values = [float(index + 1) for index in range(32)]
    forward = _norms(values)
    shuffled = dict(reversed(list(forward.items())))
    assert reference_norm(shuffled, PROMPTS) == reference_norm(forward, PROMPTS)
    assert reference_norm(forward, list(reversed(PROMPTS))) == reference_norm(forward, PROMPTS)


def test_missing_calibration_prompts_are_rejected() -> None:
    partial = _norms([1.0] * 32)
    partial.pop(PROMPTS[0])
    with pytest.raises(StrengthError, match="do not match the calibration prompts"):
        validate_calibration_norms(partial, PROMPTS)


def test_unexpected_calibration_prompts_are_rejected() -> None:
    extra = _norms([1.0] * 32) | {"intruder": 1.0}
    with pytest.raises(StrengthError, match="unexpected"):
        validate_calibration_norms(extra, PROMPTS)


def test_duplicate_expected_prompts_are_rejected() -> None:
    with pytest.raises(StrengthError, match="repeat"):
        validate_calibration_norms({"a": 1.0}, ["a", "a"])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_or_non_finite_norms_are_rejected(bad: float) -> None:
    values = [1.0] * 32
    values[7] = bad
    with pytest.raises(StrengthError):
        validate_calibration_norms(_norms(values), PROMPTS)


def test_alpha_is_the_ratio_times_the_reference_norm() -> None:
    assert alpha_for_ratio(32.0, 0.10) == pytest.approx(3.2)
    assert alpha_for_ratio(32.0, 0.02) == pytest.approx(0.64)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_alpha_rejects_a_degenerate_ratio_or_reference(bad: float) -> None:
    with pytest.raises(StrengthError):
        alpha_for_ratio(32.0, bad)
    with pytest.raises(StrengthError):
        alpha_for_ratio(bad, 0.10)


def test_the_ratio_grid_is_frozen_and_ascending() -> None:
    assert NORM_RATIOS == (0.02, 0.05, 0.10, 0.20, 0.40)
    assert list(NORM_RATIOS) == sorted(NORM_RATIOS)


def test_alpha_table_keeps_the_preregistered_order() -> None:
    table = alpha_table(32.0)
    assert [ratio for ratio, _ in table] == list(NORM_RATIOS)
    assert [alpha for _, alpha in table] == pytest.approx([0.64, 1.6, 3.2, 6.4, 12.8])


def test_alpha_table_rejects_a_reordered_grid() -> None:
    with pytest.raises(StrengthError, match="ascending"):
        alpha_table(32.0, ratios=[0.10, 0.02])


def test_one_alpha_applies_to_every_prompt() -> None:
    observations = [(prompt, 3.2) for prompt in PROMPTS]
    assert check_global_alpha(observations, layer=13, ratio=0.10) == pytest.approx(3.2)


def test_a_prompt_specific_alpha_is_rejected() -> None:
    """This is the leak the global rule exists to prevent, so it is refused, not averaged."""
    observations = [(prompt, 3.2) for prompt in PROMPTS]
    observations[5] = (PROMPTS[5], 3.3)
    with pytest.raises(StrengthError, match="more than one alpha"):
        check_global_alpha(observations, layer=13, ratio=0.10)


def test_one_deviant_alpha_is_not_hidden_by_a_repeated_label() -> None:
    """A prompt contributes many candidates; collapsing them by label would lose the deviant."""
    observations = [("g0000", 3.2), ("g0000", 99.0), ("g0000", 3.2)]
    with pytest.raises(StrengthError, match="more than one alpha"):
        check_global_alpha(observations, layer=13, ratio=0.10)


def test_there_is_no_prompt_relative_strength_function() -> None:
    """A per-prompt alpha must not be available at all, not merely discouraged."""
    from causal_self_forecasting.calibration import strength

    names = [name.lower() for name in dir(strength)]
    for banned in ("prompt_alpha", "per_prompt", "prompt_relative", "state_relative_alpha"):
        assert not any(banned in name for name in names)


# ---------------------------------------------------------------------------
# Pass conditions
# ---------------------------------------------------------------------------


def _samples(
    magnitudes: list[float],
    noop_targets: list[float] | None = None,
    flips: int = 0,
) -> list[EffectSample]:
    samples = [
        EffectSample(
            prompt_id=f"g{index:04d}",
            candidate_id=f"c{index}",
            is_noop=False,
            target=value,
            answer_flip=index < flips,
        )
        for index, value in enumerate(magnitudes)
    ]
    for index, value in enumerate(noop_targets or []):
        samples.append(
            EffectSample(
                prompt_id=f"g{index:04d}",
                candidate_id=f"noop{index}",
                is_noop=True,
                target=value,
                answer_flip=False,
            )
        )
    return samples


def _summary(
    magnitudes: list[float],
    noop_targets: list[float] | None = None,
    expected_non_noop: int | None = None,
    expected_noop: int | None = None,
    failure_count: int = 0,
    layer: int = 13,
    ratio: float = 0.10,
    flips: int = 0,
):
    noops = noop_targets if noop_targets is not None else [0.0]
    return summarize_ratio(
        samples=_samples(magnitudes, noops, flips=flips),
        layer=layer,
        norm_ratio=ratio,
        global_alpha=3.2,
        expected_non_noop=len(magnitudes) if expected_non_noop is None else expected_non_noop,
        expected_noop=len(noops) if expected_noop is None else expected_noop,
        thresholds=THRESHOLDS,
        noop_tolerance=NOOP_TOLERANCE,
        failure_count=failure_count,
    )


def _criterion(summary, name: str):
    return next(item for item in summary.criteria if item.name == name)


def test_a_healthy_ratio_passes_every_condition() -> None:
    summary = _summary([0.5] * 20, flips=3)
    assert summary.passed is True
    assert summary.flip_count == 3
    assert summary.median_abs_effect == pytest.approx(0.5)


def test_completeness_counts_recorded_failures() -> None:
    """A run missing observations passes only when every absence is written down."""
    assert _criterion(_summary([0.5] * 18, expected_non_noop=20), "completeness").passed is False
    assert (
        _criterion(
            _summary([0.5] * 18, expected_non_noop=20, failure_count=2), "completeness"
        ).passed
        is True
    )


def test_a_missing_noop_fails_completeness() -> None:
    assert _criterion(_summary([0.5] * 20, expected_noop=2), "completeness").passed is False


def test_a_non_finite_target_fails_the_finite_condition() -> None:
    summary = _summary([0.5] * 19 + [float("nan")])
    assert _criterion(summary, "finite_outputs").passed is False
    assert summary.passed is False


def test_a_noop_outside_tolerance_fails() -> None:
    assert _criterion(_summary([0.5] * 20, [0.002]), "noop_within_tolerance").passed is False
    assert _criterion(_summary([0.5] * 20, [0.0005]), "noop_within_tolerance").passed is True


def test_the_noop_condition_is_inclusive_at_the_tolerance() -> None:
    assert (
        _criterion(_summary([0.5] * 20, [NOOP_TOLERANCE]), "noop_within_tolerance").passed is True
    )


def test_the_large_effect_fraction_is_inclusive_at_fifteen_percent() -> None:
    """Exactly 3 of 20 above the threshold is 0.15 and must pass."""
    magnitudes = [0.10, 0.10, 0.10] + [0.06] * 17
    summary = _summary(magnitudes)
    assert summary.fraction_above_effect_threshold == pytest.approx(0.15)
    assert _criterion(summary, "large_effect_fraction").passed is True

    below = _summary([0.10, 0.10] + [0.06] * 18)
    assert below.fraction_above_effect_threshold == pytest.approx(0.10)
    assert _criterion(below, "large_effect_fraction").passed is False


def test_the_large_effect_threshold_is_inclusive_at_the_magnitude() -> None:
    """An effect of exactly 0.10 counts as reaching 0.10."""
    summary = _summary([0.10] * 20)
    assert summary.fraction_above_effect_threshold == pytest.approx(1.0)


def test_the_median_condition_is_inclusive_at_the_floor() -> None:
    """Two central values of 0.04 and 0.06 average to exactly 0.05."""
    magnitudes = [0.01, 0.02, 0.04, 0.06, 0.5, 0.5]
    summary = _summary(magnitudes)
    assert summary.median_abs_effect == pytest.approx(0.05)
    assert _criterion(summary, "median_abs_effect").passed is True

    lower = _summary([0.01, 0.02, 0.03, 0.05, 0.5, 0.5])
    assert lower.median_abs_effect == pytest.approx(0.04)
    assert _criterion(lower, "median_abs_effect").passed is False


def test_the_p95_condition_is_inclusive_at_the_ceiling() -> None:
    summary = _summary([4.0] * 20)
    assert summary.p95_abs_effect == pytest.approx(4.0)
    assert _criterion(summary, "p95_abs_effect").passed is True

    over = _summary([4.0] * 19 + [40.0])
    assert over.p95_abs_effect > 4.0
    assert _criterion(over, "p95_abs_effect").passed is False


def test_the_percentile_convention_is_numpy_linear() -> None:
    magnitudes = [float(index) for index in range(1, 21)]
    summary = _summary(magnitudes, expected_non_noop=20)
    expected = float(np.quantile(np.asarray(magnitudes), 0.95, method="linear"))
    assert summary.p95_abs_effect == pytest.approx(expected)


def test_the_sign_of_an_effect_does_not_matter() -> None:
    assert _summary([-0.5] * 20).median_abs_effect == pytest.approx(0.5)


def test_summarizing_nothing_is_refused() -> None:
    from causal_self_forecasting.calibration.criteria import CriteriaError

    with pytest.raises(CriteriaError, match="no observations"):
        summarize_ratio(
            samples=[],
            layer=13,
            norm_ratio=0.10,
            global_alpha=3.2,
            expected_non_noop=10,
            expected_noop=1,
            thresholds=THRESHOLDS,
            noop_tolerance=NOOP_TOLERANCE,
        )


def test_a_summary_cannot_claim_to_pass_when_a_criterion_failed() -> None:
    summary = _summary([0.5] * 20)
    dumped = summary.model_dump(mode="json")
    dumped["criteria"][0]["passed"] = False
    with pytest.raises(ValidationError, match="a ratio passes only when every condition passes"):
        type(summary).model_validate(dumped)


# ---------------------------------------------------------------------------
# The layer state machine
# ---------------------------------------------------------------------------


def _grid(passing: set[float], layer: int = 13) -> list:
    summaries = []
    for ratio in NORM_RATIOS:
        magnitudes = [0.5] * 20 if ratio in passing else [0.001] * 20
        summaries.append(_summary(magnitudes, layer=layer, ratio=ratio))
    return summaries


def test_the_smallest_passing_ratio_wins() -> None:
    selection = select_calibration_ratio(13, 20, NORM_RATIOS, _grid({0.10, 0.20, 0.40}))
    assert selection.status is CalibrationStatus.PASSED_PRIMARY
    assert selection.selected_norm_ratio == pytest.approx(0.10)
    assert selection.selected_layer == 13
    assert "smallest passing ratio" in selection.rationale


def test_selection_ignores_effect_size_and_flip_count() -> None:
    """A bigger effect at a larger ratio must not win over a smaller passing ratio.

    Ratio 0.02 barely passes: exactly 15 percent of its effects reach 0.10 and its median is
    0.06. Ratio 0.10 passes with effects forty times larger and every trial flipping. The
    smallest passing ratio still wins, because selecting on effect size would choose the
    stimulus using the outcome.
    """
    barely = [0.10, 0.10, 0.10] + [0.06] * 17
    summaries = [
        _summary(barely, ratio=0.02),
        _summary([0.001] * 20, ratio=0.05),
        _summary([3.9] * 20, ratio=0.10, flips=20),
        _summary([0.001] * 20, ratio=0.20),
        _summary([0.001] * 20, ratio=0.40),
    ]
    assert summaries[0].passed and summaries[2].passed
    assert summaries[2].median_abs_effect > summaries[0].median_abs_effect
    selection = select_calibration_ratio(13, 20, NORM_RATIOS, summaries)
    assert selection.selected_norm_ratio == pytest.approx(0.02)


def test_no_passing_primary_ratio_requires_the_fallback() -> None:
    selection = select_calibration_ratio(13, 20, NORM_RATIOS, _grid(set()))
    assert selection.status is CalibrationStatus.FALLBACK_REQUIRED
    assert selection.selected is None
    assert "only trigger" in selection.rationale


def test_a_passing_primary_prohibits_the_fallback() -> None:
    with pytest.raises(SelectionError, match="must not be calibrated"):
        select_calibration_ratio(
            13, 20, NORM_RATIOS, _grid({0.05}), fallback_summaries=_grid({0.02}, layer=20)
        )


def test_the_fallback_selects_its_own_smallest_passing_ratio() -> None:
    selection = select_calibration_ratio(
        13, 20, NORM_RATIOS, _grid(set()), fallback_summaries=_grid({0.20, 0.40}, layer=20)
    )
    assert selection.status is CalibrationStatus.PASSED_FALLBACK
    assert selection.selected_layer == 20
    assert selection.selected_norm_ratio == pytest.approx(0.20)
    assert len(selection.summaries) == 10


def test_failing_both_layers_stops_the_study() -> None:
    selection = select_calibration_ratio(
        13, 20, NORM_RATIOS, _grid(set()), fallback_summaries=_grid(set(), layer=20)
    )
    assert selection.status is CalibrationStatus.FAILED_ALL_LAYERS
    assert selection.selected is None
    assert "no third layer is searched" in selection.rationale


def test_a_partial_ratio_grid_is_refused() -> None:
    """A missing ratio could make a larger one look like the smallest that passed."""
    with pytest.raises(SelectionError, match="exactly the preregistered ratios"):
        select_calibration_ratio(13, 20, NORM_RATIOS, _grid({0.10})[:3])


def test_a_widened_ratio_grid_is_refused() -> None:
    summaries = [*_grid({0.10}), _summary([0.5] * 20, ratio=0.80)]
    with pytest.raises(SelectionError, match="unexpected"):
        select_calibration_ratio(13, 20, NORM_RATIOS, summaries)


def test_a_duplicated_ratio_is_refused() -> None:
    summaries = [*_grid({0.10}), _summary([0.5] * 20, ratio=0.10)]
    with pytest.raises(SelectionError, match="more than one summary"):
        select_calibration_ratio(13, 20, NORM_RATIOS, summaries)


def test_a_summary_for_the_wrong_layer_is_refused() -> None:
    summaries = _grid({0.10})
    summaries[0] = _summary([0.5] * 20, layer=20, ratio=0.02)
    with pytest.raises(SelectionError, match="layer 20 was supplied"):
        select_calibration_ratio(13, 20, NORM_RATIOS, summaries)


def test_the_fallback_layer_must_differ_from_the_primary() -> None:
    with pytest.raises(SelectionError, match="must differ"):
        select_calibration_ratio(13, 13, NORM_RATIOS, _grid({0.10}))


# ---------------------------------------------------------------------------
# Forward-count arithmetic
# ---------------------------------------------------------------------------


def test_the_planned_forward_counts_are_exact() -> None:
    counts = build_forward_counts(
        role_counts={"smoke": 8, "calibration": 32, "training": 96, "final_test": 32},
        direction_count=8,
        signed_directions=16,
        ratio_count=5,
    )
    assert counts.candidates_per_other_prompt == 17
    assert counts.candidates_per_calibration_prompt == 81
    assert counts.forwards_per_other_prompt == 18
    assert counts.forwards_per_calibration_prompt == 82
    assert counts.smoke_forwards == 144
    assert counts.calibration_forwards_per_layer == 2624
    assert counts.training_forwards == 1728
    assert counts.final_test_forwards == 576
    assert counts.primary_total_forwards == 5072
    assert counts.fallback_additional_forwards == 2624
    assert counts.with_fallback_total_forwards == 7696


def test_the_superseded_thirteen_thousand_figure_is_not_produced() -> None:
    """The audit's first estimate applied the ratio grid to every role. It was 2.7x too high."""
    counts = build_forward_counts(
        role_counts={"smoke": 8, "calibration": 32, "training": 96, "final_test": 32},
        direction_count=8,
        signed_directions=16,
        ratio_count=5,
    )
    assert counts.primary_total_forwards != 13776
    assert counts.primary_total_forwards == 168 * 18 + 32 * (82 - 18)


def test_forward_counts_reject_inconsistent_arithmetic() -> None:
    from causal_self_forecasting.schemas import CalibrationForwardCounts

    counts = build_forward_counts(
        role_counts={"smoke": 8, "calibration": 32, "training": 96, "final_test": 32},
        direction_count=8,
        signed_directions=16,
        ratio_count=5,
    )
    dumped = counts.model_dump(mode="json")
    dumped["primary_total_forwards"] = 5073
    with pytest.raises(ValidationError, match="does not add up"):
        CalibrationForwardCounts.model_validate(dumped)


def test_signed_directions_must_be_two_per_direction() -> None:
    with pytest.raises(CalibrationPlanError, match="two per direction"):
        build_forward_counts(
            role_counts={"smoke": 8, "calibration": 32, "training": 96, "final_test": 32},
            direction_count=8,
            signed_directions=8,
            ratio_count=5,
        )


# ---------------------------------------------------------------------------
# The plan record
# ---------------------------------------------------------------------------


REAL_CONFIG = "configs/calibration/bluedot_state_dependence.yaml"


@pytest.fixture
def plan_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    from causal_self_forecasting.calibration import plan as plan_module

    target = tmp_path / "calibration_plans"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(plan_module, "calibration_plan_path", lambda pid: target / f"{pid}.json")
    return target


@functools.lru_cache(maxsize=1)
def _real_plan_cached() -> CalibrationPlanRecord:
    """Build the real plan once. It reads two manifests and validates 168 assignments."""
    config = load_config(REAL_CONFIG, CalibrationPlanConfig)
    return build_calibration_plan(config, REAL_CONFIG)


def _real_plan() -> CalibrationPlanRecord:
    return _real_plan_cached()


def test_the_plan_cites_the_frozen_manifests() -> None:
    plan = _real_plan()
    assert plan.target_name == "delta_clean_top_margin"
    assert plan.model_id == "google/gemma-3-1b-it"
    assert plan.model_revision == "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    assert plan.calibration_prompt_count == 32
    assert plan.direction_count == 8
    assert plan.primary_layer == 13
    assert plan.fallback_layer == 20
    assert plan.norm_ratios == [0.02, 0.05, 0.10, 0.20, 0.40]
    assert plan.forward_counts.primary_total_forwards == 5072
    assert plan.scientific_result is False


def test_the_plan_hash_is_the_hash_of_its_own_payload() -> None:
    plan = _real_plan()
    assert plan.plan_hash == compute_calibration_plan_hash(plan.model_dump(mode="json"))


def test_the_plan_is_deterministic() -> None:
    assert plan_content_bytes(_real_plan()) == plan_content_bytes(_real_plan())


@pytest.mark.parametrize(
    "field,value",
    [
        ("primary_layer", 20),
        ("noop_tolerance", 1e-2),
        ("prompt_manifest_hash", "sha256:" + "0" * 64),
        ("direction_family_hash", "sha256:" + "0" * 64),
        ("direction_count", 7),
        ("master_seed", 1),
    ],
)
def test_the_plan_hash_is_sensitive_to_its_inputs(field: str, value: object) -> None:
    dumped = _real_plan().model_dump(mode="json")
    before = compute_calibration_plan_hash(dumped)
    dumped[field] = value
    assert compute_calibration_plan_hash(dumped) != before


def test_the_plan_hash_is_sensitive_to_thresholds() -> None:
    dumped = _real_plan().model_dump(mode="json")
    before = compute_calibration_plan_hash(dumped)
    dumped["thresholds"]["min_median_abs_effect"] = 0.06
    assert compute_calibration_plan_hash(dumped) != before


def test_the_plan_hash_is_sensitive_to_the_ratio_grid() -> None:
    dumped = _real_plan().model_dump(mode="json")
    before = compute_calibration_plan_hash(dumped)
    dumped["norm_ratios"] = [0.02, 0.05, 0.10, 0.20, 0.30]
    assert compute_calibration_plan_hash(dumped) != before


def test_the_plan_hash_ignores_creation_metadata() -> None:
    plan = _real_plan()
    dumped = plan.model_dump(mode="json")
    dumped["created_at"] = "2020-01-01T00:00:00Z"
    dumped["config_path"] = "elsewhere.yaml"
    assert compute_calibration_plan_hash(dumped) == plan.plan_hash


def test_the_plan_round_trips_and_rejects_tampering() -> None:
    plan = _real_plan()
    restored = CalibrationPlanRecord.model_validate(
        json.loads(json.dumps(plan.model_dump(mode="json")))
    )
    assert restored.plan_hash == plan.plan_hash

    dumped = plan.model_dump(mode="json")
    dumped["plan_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="does not match the plan contents"):
        CalibrationPlanRecord.model_validate(dumped)


def test_the_plan_rejects_a_descending_ratio_grid() -> None:
    dumped = _real_plan().model_dump(mode="json")
    dumped["norm_ratios"] = [0.40, 0.20, 0.10, 0.05, 0.02]
    dumped["plan_hash"] = compute_calibration_plan_hash(dumped)
    with pytest.raises(ValidationError, match="ascending order"):
        CalibrationPlanRecord.model_validate(dumped)


def test_writing_and_verifying_the_plan(plan_workspace: Path) -> None:
    plan = _real_plan()
    path, status = write_calibration_plan(plan)
    assert status == "written"
    assert load_calibration_plan(plan.plan_id).plan_hash == plan.plan_hash

    report = verify_calibration_plan(plan)
    assert report["valid"], report["failures"]
    assert path.exists()


def test_rewriting_an_identical_plan_preserves_bytes_and_mtime(plan_workspace: Path) -> None:
    path, _ = write_calibration_plan(_real_plan())
    original = path.read_bytes()
    mtime = path.stat().st_mtime_ns

    _, status = write_calibration_plan(_real_plan())
    assert status == "unchanged"
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == mtime


def test_a_different_plan_is_refused_at_the_same_path(plan_workspace: Path) -> None:
    plan = _real_plan()
    write_calibration_plan(plan)

    altered = plan.model_dump(mode="json")
    altered["noop_tolerance"] = 5.0e-3
    altered["plan_hash"] = compute_calibration_plan_hash(altered)
    different = CalibrationPlanRecord.model_validate(altered)

    with pytest.raises(CalibrationPlanError, match="already holds a different"):
        write_calibration_plan(different)
    _, status = write_calibration_plan(different, force=True)
    assert status == "overwritten"


# ---------------------------------------------------------------------------
# Config guards
# ---------------------------------------------------------------------------


def _config_body(**overrides) -> dict:
    body = {
        "name": "test_calibration",
        "plan_id": "test_calibration_v1",
        "study_id": "test_study",
        "target": "delta_clean_top_margin",
        "model_ref": "configs/models/gemma3_1b_it.yaml",
        "prompt_manifest_id": "bluedot_state_dependence_v1",
        "direction_family_id": "bluedot_state_dependence_directions_v1",
        "primary_layer": 13,
        "fallback_layer": 20,
        "norm_ratios": [0.02, 0.05, 0.10, 0.20, 0.40],
        "thresholds": {
            "min_large_effect_fraction": 0.15,
            "large_effect_threshold": 0.10,
            "min_median_abs_effect": 0.05,
            "max_p95_abs_effect": 4.0,
        },
        "noop_tolerance": 1.0e-3,
        "percentile_method": "numpy.quantile(method='linear')",
        "median_method": "numpy.median",
        "expected_role_counts": {"smoke": 8, "calibration": 32, "training": 96, "final_test": 32},
        "expected_direction_count": 8,
        "expected_signed_directions": 16,
        "master_seed": 20260727,
        "selection_algorithm_version": "v1",
    }
    body.update(overrides)
    return body


def test_the_real_calibration_config_validates() -> None:
    config = load_config(REAL_CONFIG, CalibrationPlanConfig)
    assert config.target == "delta_clean_top_margin"
    assert config.calibration_prompt_count == 32


def test_a_third_layer_is_rejected() -> None:
    with pytest.raises(ValidationError, match="preregistered layers"):
        CalibrationPlanConfig(**_config_body(fallback_layer=17))
    with pytest.raises(ValidationError, match="preregistered layers"):
        CalibrationPlanConfig(**_config_body(primary_layer=17))


def test_the_primary_layer_is_pinned_to_thirteen() -> None:
    with pytest.raises(ValidationError, match="primary layer is preregistered"):
        CalibrationPlanConfig(**_config_body(primary_layer=20, fallback_layer=13))


@pytest.mark.parametrize(
    "ratios",
    [
        [0.02, 0.05, 0.10, 0.20, 0.40, 0.80],
        [0.02, 0.05, 0.10, 0.20],
        [0.40, 0.20, 0.10, 0.05, 0.02],
        [0.01, 0.05, 0.10, 0.20, 0.40],
    ],
)
def test_an_altered_ratio_grid_is_rejected(ratios: list[float]) -> None:
    with pytest.raises(ValidationError, match="must be exactly"):
        CalibrationPlanConfig(**_config_body(norm_ratios=ratios))


def test_a_different_target_is_rejected() -> None:
    with pytest.raises(ValidationError, match="target must be"):
        CalibrationPlanConfig(**_config_body(target="delta_margin"))


def test_incomplete_role_counts_are_rejected() -> None:
    with pytest.raises(ValidationError, match="every prompt role"):
        CalibrationPlanConfig(**_config_body(expected_role_counts={"calibration": 32}))


def test_signed_directions_must_match_the_direction_count() -> None:
    with pytest.raises(ValidationError, match="two per direction"):
        CalibrationPlanConfig(**_config_body(expected_signed_directions=8))


def test_a_prompt_manifest_with_the_wrong_role_counts_is_refused() -> None:
    config = CalibrationPlanConfig(
        **_config_body(
            expected_role_counts={"smoke": 8, "calibration": 31, "training": 96, "final_test": 32}
        )
    )
    with pytest.raises(CalibrationPlanError, match="role counts"):
        build_calibration_plan(config, REAL_CONFIG)


def test_a_direction_family_with_the_wrong_count_is_refused() -> None:
    config = CalibrationPlanConfig(
        **_config_body(expected_direction_count=4, expected_signed_directions=8)
    )
    with pytest.raises(CalibrationPlanError, match="directions but"):
        build_calibration_plan(config, REAL_CONFIG)


def test_a_mismatched_model_revision_is_refused() -> None:
    config = CalibrationPlanConfig(**_config_body(model_ref="configs/models/gemma3_4b_it.yaml"))
    with pytest.raises(CalibrationPlanError, match=r"revision|model config is"):
        build_calibration_plan(config, REAL_CONFIG)


# ---------------------------------------------------------------------------
# The decision record
# ---------------------------------------------------------------------------


def test_the_decision_record_is_hashed_and_deterministic() -> None:
    plan = _real_plan()
    selection = select_calibration_ratio(13, 20, NORM_RATIOS, _grid({0.10}))
    first = build_decision_record(plan, selection)
    second = build_decision_record(plan, selection)
    assert first.decision_hash == second.decision_hash
    assert first.decision_hash == compute_calibration_decision_hash(first.model_dump(mode="json"))
    assert first.status is CalibrationStatus.PASSED_PRIMARY
    assert first.selected_norm_ratio == pytest.approx(0.10)
    assert first.scientific_result is False


def test_the_decision_hash_ignores_creation_metadata() -> None:
    plan = _real_plan()
    decision = build_decision_record(
        plan, select_calibration_ratio(13, 20, NORM_RATIOS, _grid({0.10}))
    )
    dumped = decision.model_dump(mode="json")
    dumped["created_at"] = "2020-01-01T00:00:00Z"
    dumped["environment"] = {"packages": {"numpy": "0.0.0"}}
    assert compute_calibration_decision_hash(dumped) == decision.decision_hash


def test_the_decision_hash_changes_with_the_selection() -> None:
    plan = _real_plan()
    first = build_decision_record(
        plan, select_calibration_ratio(13, 20, NORM_RATIOS, _grid({0.10}))
    )
    second = build_decision_record(
        plan, select_calibration_ratio(13, 20, NORM_RATIOS, _grid({0.20}))
    )
    assert first.decision_hash != second.decision_hash


def test_a_failed_decision_carries_no_selection() -> None:
    plan = _real_plan()
    decision = build_decision_record(
        plan, select_calibration_ratio(13, 20, NORM_RATIOS, _grid(set()))
    )
    assert decision.status is CalibrationStatus.FALLBACK_REQUIRED
    assert decision.selected_layer is None
    assert decision.selected_norm_ratio is None
    assert decision.selected_global_alpha is None


def test_a_decision_record_rejects_a_selection_without_a_pass() -> None:
    from causal_self_forecasting.schemas import CalibrationDecisionRecord

    plan = _real_plan()
    decision = build_decision_record(
        plan, select_calibration_ratio(13, 20, NORM_RATIOS, _grid(set()))
    )
    dumped = decision.model_dump(mode="json")
    dumped["selected_layer"] = 13
    dumped["decision_hash"] = compute_calibration_decision_hash(dumped)
    with pytest.raises(ValidationError, match="must not carry a selection"):
        CalibrationDecisionRecord.model_validate(dumped)


def test_a_tampered_decision_hash_fails_to_load() -> None:
    from causal_self_forecasting.schemas import CalibrationDecisionRecord

    plan = _real_plan()
    dumped = build_decision_record(
        plan, select_calibration_ratio(13, 20, NORM_RATIOS, _grid({0.10}))
    ).model_dump(mode="json")
    dumped["decision_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="does not match the decision contents"):
        CalibrationDecisionRecord.model_validate(dumped)
