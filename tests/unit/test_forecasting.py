"""Tests for the baselines and the leakage fence around them.

The baselines are tested on hand-built training examples, no model or run required. The
leakage tests are the point of this file: a prompt-only baseline that quietly read a hidden
state or a true mechanism would invalidate the comparison it exists to anchor, so the fence is
checked at the type level and the interface level.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from causal_self_forecasting.forecasting import (
    ConstantBaseline,
    PromptLexicalBaseline,
    TrainingExample,
)
from causal_self_forecasting.forecasting.base import (
    FORBIDDEN_FIELDS,
    PUBLIC_FEATURE_KEYS,
    CandidatePrediction,
    LeakageError,
)
from causal_self_forecasting.schemas import StateCondition


def _example(
    *,
    prompt: str = "answer the question A B C D",
    operation: str = "residual_add",
    layer: int = 2,
    strength: float = 1.0,
    delta: float = 0.0,
    flip: bool = False,
    group: str = "g0",
    split: str = "train",
) -> TrainingExample:
    return TrainingExample(
        prompt_text=prompt,
        public_features={
            "operation": operation,
            "layer": layer,
            "position_index": -1,
            "strength": strength,
        },
        observed_delta=delta,
        observed_flip=flip,
        group_id=group,
        split=split,
    )


# ---------------------------------------------------------------------------
# Leakage fence
# ---------------------------------------------------------------------------


def test_training_example_carries_no_forbidden_field() -> None:
    names = {field.name for field in fields(TrainingExample)}
    assert not (names & FORBIDDEN_FIELDS), "TrainingExample must not carry private information"


def test_public_features_are_the_only_conditioning_keys() -> None:
    example = _example()
    assert set(example.public_features) == set(PUBLIC_FEATURE_KEYS)
    for forbidden in ("mechanism", "direction_id", "analysis_role", "state_id"):
        assert forbidden not in example.public_features


def test_constant_baseline_declares_no_forbidden_inputs() -> None:
    baseline = ConstantBaseline()
    baseline.check_no_leakage()  # must not raise
    declared = baseline.declared_inputs()
    assert declared["hidden_state"] is False
    assert declared["adapter_identity"] is False
    assert declared["observed_outcomes"] is False


def test_prompt_lexical_declares_no_forbidden_inputs() -> None:
    baseline = PromptLexicalBaseline()
    baseline.check_no_leakage()  # must not raise
    declared = baseline.declared_inputs()
    assert declared["prompt"] is True
    assert declared["hidden_state"] is False
    assert declared["intervention_vector"] is False


def test_check_no_leakage_rejects_a_forbidden_declaration() -> None:
    class LeakyBaseline(ConstantBaseline):
        def declared_inputs(self) -> dict[str, bool]:
            base = super().declared_inputs()
            base["hidden_state"] = True
            return base

    with pytest.raises(LeakageError, match="hidden_state"):
        LeakyBaseline().check_no_leakage()


def test_baselines_report_state_condition_none() -> None:
    assert ConstantBaseline().state_condition() is StateCondition.NONE
    assert PromptLexicalBaseline().state_condition() is StateCondition.NONE


# ---------------------------------------------------------------------------
# Constant baseline
# ---------------------------------------------------------------------------


def test_constant_baseline_predicts_group_mean() -> None:
    baseline = ConstantBaseline()
    baseline.fit(
        [
            _example(strength=1.0, delta=2.0),
            _example(strength=1.0, delta=4.0),
            _example(strength=-1.0, delta=-3.0),
        ]
    )
    positive = baseline.predict_candidate(
        "prompt", "t.opaque_00", {"operation": "residual_add", "layer": 2, "strength": 1.0}
    )
    assert positive.delta_margin_mean == pytest.approx(3.0)
    negative = baseline.predict_candidate(
        "prompt", "t.opaque_01", {"operation": "residual_add", "layer": 2, "strength": -1.0}
    )
    assert negative.delta_margin_mean == pytest.approx(-3.0)


def test_constant_baseline_cannot_separate_random_from_real() -> None:
    """A real steer and a matched random control share a public group, by design.

    They are both a positive-strength residual addition in the public view, so the constant
    baseline gives them the same prediction. Distinguishing them is exactly what a
    state-aware method must do and this baseline cannot.
    """
    baseline = ConstantBaseline()
    baseline.fit([_example(strength=1.0, delta=2.0), _example(strength=1.0, delta=0.0)])
    real = baseline.predict_candidate(
        "p", "t.opaque_00", {"operation": "residual_add", "layer": 2, "strength": 1.0}
    )
    control = baseline.predict_candidate(
        "p", "t.opaque_01", {"operation": "residual_add", "layer": 2, "strength": 1.0}
    )
    assert real.delta_margin_mean == control.delta_margin_mean


def test_constant_baseline_falls_back_to_global_mean_for_unseen_group() -> None:
    baseline = ConstantBaseline()
    baseline.fit([_example(strength=1.0, delta=2.0), _example(strength=-1.0, delta=-2.0)])
    unseen = baseline.predict_candidate(
        "p", "t.opaque_00", {"operation": "residual_add", "layer": 99, "strength": 5.0}
    )
    assert unseen.delta_margin_mean == pytest.approx(0.0)


def test_constant_baseline_predicts_flip_rate() -> None:
    baseline = ConstantBaseline()
    baseline.fit(
        [
            _example(strength=1.0, flip=True),
            _example(strength=1.0, flip=True),
            _example(strength=1.0, flip=False),
        ]
    )
    prediction = baseline.predict_candidate(
        "p", "t.opaque_00", {"operation": "residual_add", "layer": 2, "strength": 1.0}
    )
    assert prediction.p_answer_flip == pytest.approx(2 / 3)


def test_constant_baseline_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        ConstantBaseline().predict_candidate(
            "p", "t.opaque_00", {"operation": "residual_add", "layer": 2, "strength": 1.0}
        )


# ---------------------------------------------------------------------------
# Prompt-only lexical baseline
# ---------------------------------------------------------------------------


def test_prompt_lexical_fits_and_predicts() -> None:
    baseline = PromptLexicalBaseline()
    examples = [
        _example(prompt="water energy A B C D", strength=1.0, delta=2.0, flip=True),
        _example(prompt="rock heat A B C D", strength=1.0, delta=1.8, flip=True),
        _example(prompt="light gas A B C D", strength=-1.0, delta=-2.0, flip=False),
        _example(prompt="cell force A B C D", strength=-1.0, delta=-1.9, flip=False),
    ]
    baseline.fit(examples)
    prediction = baseline.predict_candidate(
        "water energy A B C D",
        "t.opaque_00",
        {"operation": "residual_add", "layer": 2, "strength": 1.0},
    )
    assert isinstance(prediction, CandidatePrediction)
    assert 0.0 <= prediction.p_answer_flip <= 1.0
    assert (
        prediction.delta_margin_q05 <= prediction.delta_margin_mean <= prediction.delta_margin_q95
    )


def test_prompt_lexical_handles_single_flip_class() -> None:
    baseline = PromptLexicalBaseline()
    baseline.fit(
        [
            _example(prompt="water energy rock", flip=False, delta=0.1),
            _example(prompt="light gas cell", flip=False, delta=0.2),
        ]
    )
    prediction = baseline.predict_candidate(
        "water energy rock",
        "t.opaque_00",
        {"operation": "residual_add", "layer": 2, "strength": 1.0},
    )
    assert prediction.p_answer_flip == pytest.approx(0.0)


def test_prompt_lexical_requires_examples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PromptLexicalBaseline().fit([])


def test_prompt_lexical_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        PromptLexicalBaseline().predict_candidate(
            "p", "t.opaque_00", {"operation": "residual_add", "layer": 2, "strength": 1.0}
        )


def test_prediction_clamps_probability_into_range() -> None:
    prediction = CandidatePrediction(
        intervention_id="t.opaque_00",
        delta_margin_mean=0.0,
        delta_margin_q05=-1.0,
        delta_margin_q95=1.0,
        p_answer_flip=1.5,
    ).to_forecast_candidate()
    assert prediction.p_answer_flip == 1.0
    assert prediction.p_bias_suppressed == 0.5
