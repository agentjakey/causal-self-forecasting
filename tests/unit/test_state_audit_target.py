"""Tests for the study target `delta_clean_top_margin`.

The target is a small formula, which is exactly why it needs pinning: a sign slip or a
recomputed argmax would produce plausible numbers that answer a different question, and nothing
downstream would notice.

These also guard the boundary between the two targets. The benchmark's `delta_margin` measures
the margin around the dataset-correct answer and must keep doing so.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from causal_self_forecasting.schemas import (
    ObservationRecord,
    PromptRole,
    StateAuditObservationRecord,
)
from causal_self_forecasting.state_audit_target import (
    TARGET_NAME,
    TargetError,
    logits_from_pairs,
    preferred_label,
    preferred_margin,
    state_audit_target,
    validate_answer_logits,
    verify_state_audit_target,
)

CLEAN = {"A": 3.0, "B": 1.0, "C": 0.5, "D": 0.0}


# ---------------------------------------------------------------------------
# Preferred label and tie-breaking
# ---------------------------------------------------------------------------


def test_preferred_label_is_the_argmax() -> None:
    assert preferred_label(CLEAN) == "A"
    assert preferred_label({"A": 0.0, "B": 0.0, "C": 9.0, "D": 1.0}) == "C"


@pytest.mark.parametrize(
    "logits,expected",
    [
        ({"A": 1.0, "B": 1.0, "C": 0.0, "D": 0.0}, "A"),
        ({"A": 0.0, "B": 1.0, "C": 1.0, "D": 0.0}, "B"),
        ({"A": 0.0, "B": 0.0, "C": 1.0, "D": 1.0}, "C"),
        ({"A": 2.0, "B": 2.0, "C": 2.0, "D": 2.0}, "A"),
    ],
)
def test_exact_ties_break_in_fixed_label_order(logits: dict[str, float], expected: str) -> None:
    """A tie must resolve the same way on every machine and in every rerun."""
    assert preferred_label(logits) == expected


def test_preferred_margin_uses_the_label_it_is_given() -> None:
    """The intervened margin is measured against the clean preference, not a fresh argmax."""
    assert preferred_margin(CLEAN, "A") == pytest.approx(2.0)
    assert preferred_margin(CLEAN, "B") == pytest.approx(-2.0)
    assert preferred_margin(CLEAN, "D") == pytest.approx(-3.0)


def test_preferred_margin_rejects_an_unknown_label() -> None:
    with pytest.raises(TargetError, match="not one of"):
        preferred_margin(CLEAN, "E")


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------


def test_target_is_the_change_in_the_clean_preferred_margin() -> None:
    intervened = {"A": 2.0, "B": 1.5, "C": 0.5, "D": 0.0}
    result = state_audit_target(CLEAN, intervened)
    assert result.clean_preferred_label == "A"
    assert result.clean_top_margin == pytest.approx(2.0)
    assert result.intervened_top_margin == pytest.approx(0.5)
    assert result.delta_clean_top_margin == pytest.approx(-1.5)
    assert result.answer_flip is False


def test_c_star_is_held_fixed_after_the_intervention() -> None:
    """A recomputed argmax would make the intervened margin non-negative and hide every flip."""
    intervened = {"A": 0.0, "B": 5.0, "C": 0.0, "D": 0.0}
    result = state_audit_target(CLEAN, intervened)
    assert result.clean_preferred_label == "A"
    assert result.intervened_top_margin == pytest.approx(-5.0)
    assert result.delta_clean_top_margin == pytest.approx(-7.0)
    assert result.answer_flip is True
    assert result.intervened_preferred_label == "B"


def test_clean_top_margin_is_never_negative() -> None:
    for logits in (
        {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0},
        {"A": -5.0, "B": -9.0, "C": -7.0, "D": -8.0},
    ):
        assert state_audit_target(logits, logits).clean_top_margin >= 0.0


def test_an_unchanged_output_gives_a_zero_target() -> None:
    result = state_audit_target(CLEAN, dict(CLEAN))
    assert result.delta_clean_top_margin == 0.0
    assert result.answer_flip is False


def test_a_negative_intervened_margin_always_means_a_flip() -> None:
    result = state_audit_target(CLEAN, {"A": 1.0, "B": 4.0, "C": 0.0, "D": 0.0})
    assert result.intervened_top_margin < 0.0
    assert result.answer_flip is True


def test_flip_at_an_exact_tie_follows_the_label_order() -> None:
    """The one place where flip and `margin < 0` can disagree, pinned rather than assumed."""
    clean = {"A": 0.0, "B": 3.0, "C": 0.0, "D": 0.0}
    assert preferred_label(clean) == "B"
    # A ties B after the intervention. A comes first in label order, so the top label moves.
    tied = {"A": 2.0, "B": 2.0, "C": 0.0, "D": 0.0}
    result = state_audit_target(clean, tied)
    assert result.intervened_top_margin == 0.0
    assert result.answer_flip is True
    assert result.intervened_preferred_label == "A"

    # The mirror case: c_star wins the tie, so there is no flip at the same zero margin.
    clean_a = {"A": 3.0, "B": 0.0, "C": 0.0, "D": 0.0}
    result_a = state_audit_target(clean_a, tied)
    assert result_a.intervened_top_margin == 0.0
    assert result_a.answer_flip is False


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "logits",
    [
        {"A": 1.0, "B": 1.0, "C": 1.0},
        {"A": 1.0, "B": 1.0, "C": 1.0, "E": 1.0},
        {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0},
        {},
    ],
)
def test_logit_vectors_must_be_exactly_the_four_labels(logits: dict[str, float]) -> None:
    with pytest.raises(TargetError, match="exactly"):
        validate_answer_logits(logits)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_logits_are_rejected(bad: float) -> None:
    logits = dict(CLEAN) | {"C": bad}
    with pytest.raises(TargetError, match="finite"):
        validate_answer_logits(logits)


def test_non_numeric_logits_are_rejected() -> None:
    with pytest.raises(TargetError, match="not a real number"):
        validate_answer_logits({"A": 1.0, "B": "2.0", "C": 0.0, "D": 0.0})  # type: ignore[dict-item]


def test_duplicate_labels_are_rejected_when_building_from_pairs() -> None:
    with pytest.raises(TargetError, match="more than once"):
        logits_from_pairs([("A", 1.0), ("A", 2.0), ("B", 0.0), ("C", 0.0), ("D", 0.0)])


def test_pairs_build_a_valid_mapping() -> None:
    built = logits_from_pairs([("A", 1.0), ("B", 2.0), ("C", 3.0), ("D", 4.0)])
    assert built == {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}


# ---------------------------------------------------------------------------
# Independent verification
# ---------------------------------------------------------------------------


def test_verification_accepts_a_consistent_target() -> None:
    intervened = {"A": 2.0, "B": 1.5, "C": 0.5, "D": 0.0}
    result = state_audit_target(CLEAN, intervened)
    assert (
        verify_state_audit_target(
            CLEAN,
            intervened,
            result.clean_preferred_label,
            result.clean_top_margin,
            result.intervened_top_margin,
            result.delta_clean_top_margin,
            result.answer_flip,
        )
        == []
    )


def test_verification_reports_every_disagreement() -> None:
    intervened = {"A": 2.0, "B": 1.5, "C": 0.5, "D": 0.0}
    problems = verify_state_audit_target(CLEAN, intervened, "B", 99.0, 0.5, 0.0, True)
    assert len(problems) == 4
    assert any("clean_preferred_label" in problem for problem in problems)
    assert any("clean_top_margin" in problem for problem in problems)
    assert any(TARGET_NAME in problem for problem in problems)
    assert any("answer_flip" in problem for problem in problems)


def test_verification_rejects_a_non_finite_stored_value() -> None:
    problems = verify_state_audit_target(CLEAN, dict(CLEAN), "A", math.nan, 2.0, 0.0, False)
    assert any("not finite" in problem for problem in problems)


# ---------------------------------------------------------------------------
# The state-audit record
# ---------------------------------------------------------------------------


def _record(**overrides) -> StateAuditObservationRecord:
    clean = dict(CLEAN)
    intervened = {"A": 2.0, "B": 1.5, "C": 0.5, "D": 0.0}
    computed = state_audit_target(clean, intervened)
    body = {
        "study_id": "bluedot_state_dependence",
        "run_id": "cal-run",
        "trial_id": "trial_00001",
        "candidate_id": "trial_00001.opaque_03",
        "group_id": "g0001",
        "variant_id": "g0001.neutral_a",
        "prompt_role": PromptRole.CALIBRATION,
        "clean_preferred_label": computed.clean_preferred_label,
        "clean_logits": clean,
        "intervened_logits": intervened,
        "clean_top_margin": computed.clean_top_margin,
        "intervened_top_margin": computed.intervened_top_margin,
        "delta_clean_top_margin": computed.delta_clean_top_margin,
        "answer_flip": computed.answer_flip,
        "is_noop": False,
        "direction_ref": "bd1.1dd98ed52e4a9a42",
        "layer": 13,
        "norm_ratio": 0.10,
        "global_alpha": 3.25,
        "model_id": "google/gemma-3-1b-it",
        "model_revision": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "prompt_manifest_hash": "sha256:" + "1" * 64,
        "direction_family_hash": "sha256:" + "2" * 64,
        "config_hash": "sha256:" + "3" * 64,
    }
    body.update(overrides)
    return StateAuditObservationRecord(**body)  # type: ignore[arg-type]


def test_state_audit_record_round_trips() -> None:
    import json

    record = _record()
    restored = StateAuditObservationRecord.model_validate(
        json.loads(json.dumps(record.model_dump(mode="json")))
    )
    assert restored == record
    assert restored.target_name == TARGET_NAME


def test_state_audit_record_rejects_an_inconsistent_stored_target() -> None:
    with pytest.raises(ValidationError, match="disagrees with its own"):
        _record(delta_clean_top_margin=0.0)


def test_state_audit_record_rejects_a_wrong_preferred_label() -> None:
    with pytest.raises(ValidationError, match="disagrees with its own"):
        _record(clean_preferred_label="D")


def test_state_audit_record_rejects_a_wrong_flip() -> None:
    with pytest.raises(ValidationError, match="disagrees with its own"):
        _record(answer_flip=True)


def test_state_audit_record_rejects_non_finite_logits() -> None:
    with pytest.raises(ValidationError, match="finite"):
        _record(intervened_logits={"A": 1.0, "B": float("inf"), "C": 0.0, "D": 0.0})


def test_state_audit_record_rejects_missing_labels() -> None:
    with pytest.raises(ValidationError, match="exactly"):
        _record(clean_logits={"A": 1.0, "B": 0.0, "C": 0.0})


@pytest.mark.parametrize(
    "reference",
    [
        "answer_token_centered_00",
        "random_orthogonal_control_2",
        "bd1.direction_positive",
        "family.random_control",
        "x.construction_role",
    ],
)
def test_state_audit_record_rejects_a_semantic_direction_reference(reference: str) -> None:
    """A direction reference that names its role would hand a reader the family for free."""
    with pytest.raises(ValidationError, match="opaque"):
        _record(direction_ref=reference)


def test_state_audit_record_requires_a_zero_alpha_for_a_noop() -> None:
    with pytest.raises(ValidationError, match="no-op observation"):
        _record(is_noop=True, norm_ratio=0.10, global_alpha=3.25)


def test_a_noop_record_is_accepted_with_zero_strength() -> None:
    record = _record(
        is_noop=True,
        norm_ratio=0.0,
        global_alpha=0.0,
        intervened_logits=dict(CLEAN),
        clean_top_margin=2.0,
        intervened_top_margin=2.0,
        delta_clean_top_margin=0.0,
        answer_flip=False,
    )
    assert record.is_noop is True
    assert record.delta_clean_top_margin == 0.0


def test_a_non_noop_record_requires_a_nonzero_alpha() -> None:
    with pytest.raises(ValidationError, match="zero global_alpha"):
        _record(global_alpha=0.0)


# ---------------------------------------------------------------------------
# The original benchmark record is untouched
# ---------------------------------------------------------------------------


def test_the_original_observation_record_still_validates_its_own_delta() -> None:
    """`ObservationRecord` keeps its dataset-correct-answer semantics and its own validator."""
    body = {
        "trial_id": "trial_00001",
        "intervention_id": "trial_00001.opaque_00",
        "mechanism": "residual_add",
        "clean_logits": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
        "post_logits": {"A": 2.0, "B": 0.0, "C": 0.0, "D": 0.0},
        "clean_margin": 1.0,
        "post_margin": 2.0,
        "delta_margin": 1.0,
        "clean_predicted_label": "A",
        "post_predicted_label": "A",
        "answer_flip": False,
        "clean_entropy": 0.5,
        "post_entropy": 0.4,
        "pre_norm": 1.0,
        "post_norm": 1.2,
    }
    record = ObservationRecord(**body)  # type: ignore[arg-type]
    assert record.delta_margin == pytest.approx(1.0)

    with pytest.raises(ValidationError, match="does not match post_margin minus clean_margin"):
        ObservationRecord(**(body | {"delta_margin": 5.0}))  # type: ignore[arg-type]


def test_the_two_records_are_distinct_types() -> None:
    assert StateAuditObservationRecord is not ObservationRecord
    assert "delta_clean_top_margin" not in ObservationRecord.model_fields
    assert "delta_margin" not in StateAuditObservationRecord.model_fields
