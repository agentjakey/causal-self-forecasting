"""Tests for the record schemas.

The validators here are the ones that stop an incoherent artifact from ever reaching disk,
where it would later be read back as evidence.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from causal_self_forecasting.schemas import (
    ForecastCandidate,
    InterventionSpec,
    Mechanism,
    MetricValue,
    ModelSpec,
    ModelVariant,
    ObservationRecord,
    PublicDashboardRecord,
    ResultStatus,
    Split,
    TaskItem,
)


def _task_item(**overrides) -> dict:
    base = dict(
        item_id="q1",
        group_id="q1",
        source="allenai/ai2_arc",
        source_id="Mercury_1",
        subject="ARC-Challenge",
        question="What is water?",
        choices=["a liquid", "a gas", "a solid", "a plasma"],
        answer_index=0,
        split=Split.TRAIN,
    )
    base.update(overrides)
    return base


def test_task_item_exposes_its_answer_label() -> None:
    assert TaskItem(**_task_item(answer_index=2)).answer_label == "C"


def test_task_item_rejects_duplicate_choices() -> None:
    """Duplicate options make the correct answer ambiguous."""
    with pytest.raises(ValidationError, match="duplicate choices"):
        TaskItem(**_task_item(choices=["a", "a", "b", "c"]))


def test_task_item_rejects_an_empty_choice() -> None:
    with pytest.raises(ValidationError, match="empty choice"):
        TaskItem(**_task_item(choices=["a", "   ", "b", "c"]))


def test_task_item_requires_exactly_four_choices() -> None:
    with pytest.raises(ValidationError):
        TaskItem(**_task_item(choices=["a", "b", "c"]))


def test_model_spec_rejects_a_moving_revision() -> None:
    """A run pinned to `main` cannot be reproduced once upstream changes."""
    with pytest.raises(ValidationError, match="moving pointer"):
        ModelSpec(model_id="google/gemma-3-1b-it", revision="main")


def test_model_spec_requires_an_adapter_for_the_adapted_variant() -> None:
    with pytest.raises(ValidationError, match="requires adapter_path"):
        ModelSpec(model_id="m", revision="abc123", variant=ModelVariant.ADAPTED)


def test_model_spec_rejects_an_adapter_on_the_clean_variant() -> None:
    """Guards against a state captured with an adapter attached being labeled clean."""
    with pytest.raises(ValidationError, match="must not carry an adapter"):
        ModelSpec(model_id="m", revision="abc123", variant=ModelVariant.CLEAN, adapter_path="a/b")


def _spec(**overrides) -> dict:
    base = dict(
        intervention_id="t1.opaque_00",
        mechanism=Mechanism.RESIDUAL_ADD,
        mechanism_version="1.0",
        layer=10,
        position_index=-1,
        strength=1.0,
        direction_id="d1",
    )
    base.update(overrides)
    return base


def test_intervention_requires_a_direction_for_additive_mechanisms() -> None:
    with pytest.raises(ValidationError, match="requires direction_id"):
        InterventionSpec(**_spec(direction_id=None))


def test_intervention_requires_a_source_state_for_patching() -> None:
    with pytest.raises(ValidationError, match="requires source_state_id"):
        InterventionSpec(**_spec(mechanism=Mechanism.ACTIVATION_PATCH, direction_id=None))


def test_noop_must_have_zero_strength() -> None:
    with pytest.raises(ValidationError, match="must have strength 0"):
        InterventionSpec(**_spec(mechanism=Mechanism.NOOP, strength=1.0, direction_id=None))


def test_ablation_is_only_defined_at_full_strength() -> None:
    with pytest.raises(ValidationError, match=re.escape("strength 1.0")):
        InterventionSpec(**_spec(mechanism=Mechanism.DIRECTION_ABLATE, strength=0.5))


@pytest.mark.parametrize("key", ["role", "analysis_role", "direction_id", "answer", "delta_margin"])
def test_public_metadata_cannot_carry_leaking_keys(key: str) -> None:
    """`public_metadata` is shown to forecasters, so what may live in it is enumerated."""
    with pytest.raises(ValidationError, match="would leak"):
        InterventionSpec(**_spec(public_metadata={key: "x"}))


def test_forecast_candidate_rejects_an_inverted_interval() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        ForecastCandidate(
            intervention_id="t1.opaque_00",
            delta_margin_mean=0.0,
            delta_margin_q05=1.0,
            delta_margin_q95=-1.0,
            p_answer_flip=0.5,
            p_bias_suppressed=0.5,
        )


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_forecast_candidate_bounds_probabilities(probability: float) -> None:
    with pytest.raises(ValidationError):
        ForecastCandidate(
            intervention_id="t1.opaque_00",
            delta_margin_mean=0.0,
            delta_margin_q05=-1.0,
            delta_margin_q95=1.0,
            p_answer_flip=probability,
            p_bias_suppressed=0.5,
        )


def _observation(**overrides) -> dict:
    base = dict(
        trial_id="trial_00001",
        intervention_id="trial_00001.opaque_00",
        mechanism=Mechanism.RESIDUAL_ADD,
        clean_logits={"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
        post_logits={"A": 0.5, "B": 0.0, "C": 0.0, "D": 0.0},
        clean_margin=1.0,
        post_margin=0.5,
        delta_margin=-0.5,
        clean_predicted_label="A",
        post_predicted_label="A",
        answer_flip=False,
        clean_entropy=0.5,
        post_entropy=0.6,
        pre_norm=10.0,
        post_norm=10.5,
    )
    base.update(overrides)
    return base


def test_observation_accepts_consistent_fields() -> None:
    assert ObservationRecord(**_observation()).delta_margin == -0.5


def test_observation_rejects_an_inconsistent_delta() -> None:
    """delta_margin is derived, so a mismatch means something upstream is wrong."""
    with pytest.raises(ValidationError, match="does not match"):
        ObservationRecord(**_observation(delta_margin=-0.9))


def test_observation_rejects_a_flip_flag_that_contradicts_the_labels() -> None:
    with pytest.raises(ValidationError, match="disagrees"):
        ObservationRecord(**_observation(answer_flip=True))


def test_metric_value_requires_a_coherent_interval() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        MetricValue(value=0.5, n=10, ci_low=0.9, ci_high=0.1)


def test_metric_value_requires_a_positive_sample_count() -> None:
    """A point estimate with no sample count behind it is not a result."""
    with pytest.raises(ValidationError):
        MetricValue(value=0.5, n=0, ci_low=0.1, ci_high=0.9)


def test_public_export_refuses_unverified_runs() -> None:
    """The last gate before a number reaches the dashboard."""
    with pytest.raises(ValidationError, match="did not verify"):
        PublicDashboardRecord(
            run_id="run-1",
            status=ResultStatus.PRELIMINARY,
            commitments_verified=False,
        )


def test_records_reject_unknown_fields() -> None:
    """Stops a renamed or stale field from silently becoming a no-op.

    Goes through model_validate because passing an unknown keyword directly is a type error
    that never reaches runtime; what matters here is a dict read back off disk.
    """
    with pytest.raises(ValidationError):
        ModelSpec.model_validate({"model_id": "m", "revision": "abc123", "typo_field": 1})
