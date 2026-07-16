"""Tests for split assignment and candidate-set construction.

Split leakage and label shortcuts are the two failure modes that would invalidate the
headline comparison while leaving every number looking healthy.
"""

from __future__ import annotations

import pytest

from causal_self_forecasting.config import SplitConfig
from causal_self_forecasting.schemas import Mechanism, Split
from causal_self_forecasting.tasks.splitting import assign_split
from causal_self_forecasting.trials.candidates import (
    CandidateTemplate,
    build_candidate_set,
    default_templates,
    public_candidate_views,
    public_view,
)


def _config(**overrides) -> SplitConfig:
    return SplitConfig(**overrides)


def test_split_assignment_is_deterministic() -> None:
    config = _config()
    assert assign_split("q1", "science", config) == assign_split("q1", "science", config)


def test_split_assignment_does_not_depend_on_dataset_size_or_order() -> None:
    """A --max-items smoke run must not place an item differently from a full run."""
    config = _config()
    first_pass = {f"q{index}": assign_split(f"q{index}", "science", config) for index in range(50)}
    second_pass = {
        f"q{index}": assign_split(f"q{index}", "science", config) for index in reversed(range(50))
    }
    assert first_pass == second_pass


def test_held_out_subjects_never_land_in_training() -> None:
    config = _config(heldout_subjects=["professional_law"])
    for index in range(200):
        assert assign_split(f"q{index}", "professional_law", config) == Split.HELDOUT_SUBJECT


def test_subject_holdout_beats_the_random_assignment() -> None:
    without = _config()
    with_holdout = _config(heldout_subjects=["philosophy"])
    training = [
        f"q{i}" for i in range(200) if assign_split(f"q{i}", "philosophy", without) == Split.TRAIN
    ]
    assert training, "expected some items to be training items without the holdout"
    for group_id in training:
        assert assign_split(group_id, "philosophy", with_holdout) == Split.HELDOUT_SUBJECT


def test_split_fractions_are_roughly_respected() -> None:
    config = _config(train_fraction=0.6, val_fraction=0.15, test_fraction=0.25)
    assignments = [assign_split(f"q{index}", "science", config) for index in range(4000)]
    train_share = sum(1 for split in assignments if split is Split.TRAIN) / len(assignments)
    assert 0.57 < train_share < 0.63


def test_changing_the_split_seed_changes_the_partition() -> None:
    first = [assign_split(f"q{i}", "science", _config(split_seed=1)) for i in range(200)]
    second = [assign_split(f"q{i}", "science", _config(split_seed=2)) for i in range(200)]
    assert first != second


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        SplitConfig(train_fraction=0.6, val_fraction=0.3, test_fraction=0.3)


def _templates() -> list[CandidateTemplate]:
    return default_templates(
        "bias_dir", "bias_dir__random_00", layer=10, position_index=-1, strength=2.0
    )


def test_default_candidate_set_has_the_four_roles() -> None:
    roles = {template.role for template in _templates()}
    assert roles == {"direction_positive", "direction_negative", "random_control", "noop_control"}


def test_positive_and_negative_candidates_are_opposite() -> None:
    strengths = {template.role: template.strength for template in _templates()}
    assert strengths["direction_positive"] == -strengths["direction_negative"]


def test_random_control_matches_the_positive_magnitude() -> None:
    """A control at a different strength would confound magnitude with direction."""
    by_role = {template.role: template for template in _templates()}
    assert by_role["random_control"].strength == by_role["direction_positive"].strength


def test_candidate_ids_are_opaque() -> None:
    """A meaningful id would let a text forecaster score well by reading the label."""
    candidate_set = build_candidate_set("trial_00001", 12345, _templates())
    for candidate in candidate_set.candidates:
        identifier = candidate.intervention_id.lower()
        for leak in ("bias", "random", "noop", "positive", "negative", "control"):
            assert leak not in identifier


def test_candidate_order_is_seeded_and_reproducible() -> None:
    first = build_candidate_set("trial_00001", 12345, _templates())
    second = build_candidate_set("trial_00001", 12345, _templates())
    assert [c.analysis_role for c in first.candidates] == [
        c.analysis_role for c in second.candidates
    ]


def test_candidate_order_varies_across_trials() -> None:
    orders = {
        tuple(
            c.analysis_role
            for c in build_candidate_set(f"trial_{i:05d}", 1, _templates()).candidates
        )
        for i in range(40)
    }
    assert len(orders) > 1


def test_candidate_position_does_not_encode_role() -> None:
    """Ids are assigned after the shuffle, so position must carry no information."""
    roles_at_zero = {
        build_candidate_set(f"trial_{i:05d}", 7, _templates()).candidates[0].analysis_role
        for i in range(40)
    }
    assert len(roles_at_zero) > 1


def test_public_view_hides_the_direction() -> None:
    candidate_set = build_candidate_set("trial_00001", 12345, _templates())
    for view in public_candidate_views(candidate_set):
        assert "direction_id" not in view
        assert "analysis_role" not in view


def test_public_view_makes_the_random_control_indistinguishable() -> None:
    """If the public view said `random_add`, the control would stop being a control: a
    forecaster could call it a non-event without looking at the model's state at all."""
    candidate_set = build_candidate_set("trial_00001", 12345, _templates())
    by_role = {candidate.analysis_role: candidate for candidate in candidate_set.candidates}

    steer = public_view(by_role["direction_positive"])
    control = public_view(by_role["random_control"])

    assert steer["operation"] == control["operation"] == "residual_add"
    assert steer["strength"] == control["strength"]
    # The only difference is the opaque id.
    assert {k: v for k, v in steer.items() if k != "intervention_id"} == {
        k: v for k, v in control.items() if k != "intervention_id"
    }


def test_public_view_never_names_the_mechanism_enum() -> None:
    candidate_set = build_candidate_set("trial_00001", 12345, _templates())
    for candidate, view in zip(
        candidate_set.candidates, public_candidate_views(candidate_set), strict=True
    ):
        assert "mechanism" not in view
        if candidate.mechanism is Mechanism.RANDOM_ADD:
            assert view["operation"] != "random_add"


def test_noop_is_published_as_a_zero_strength_addition() -> None:
    candidate_set = build_candidate_set("trial_00001", 12345, _templates())
    by_role = {candidate.analysis_role: candidate for candidate in candidate_set.candidates}
    view = public_view(by_role["noop_control"])
    assert view["operation"] == "residual_add"
    assert view["strength"] == 0.0


def test_build_candidate_set_requires_at_least_two_candidates() -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_candidate_set("trial_00001", 1, _templates()[:1])
