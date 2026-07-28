"""Unit tests for the study-specific candidate builders.

No model, no prompts, no forward passes. Everything here is arithmetic and record validation
over hand-built direction references.

The privacy tests are the important ones. A candidate that named its construction role would
let any language-model forecaster score well by reading the label, so the record type refuses
such an id rather than relying on the builder to avoid one.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from causal_self_forecasting.hashing import canonical_json_bytes
from causal_self_forecasting.schemas import (
    Mechanism,
    PromptRole,
    StateAuditCandidate,
    StateAuditCandidateKind,
    StateAuditCandidateSet,
    compute_state_audit_candidate_set_hash,
)
from causal_self_forecasting.state_audit.candidates import (
    CandidateBuildError,
    DirectionRef,
    build_state_audit_candidate_set,
    calibration_grid_templates,
    candidate_analysis_role,
    candidate_mechanism,
    candidate_order_seed,
    public_candidate_views,
    public_view,
    selected_strength_templates,
    signed_intervention_hash,
)

RATIOS = (0.02, 0.05, 0.10, 0.20, 0.40)
REFERENCE_NORM = 32.0
FAMILY_HASH = "sha256:" + "a" * 64


def directions(count: int = 8) -> list[DirectionRef]:
    return [
        DirectionRef(opaque_id=f"bd1.{index:016x}", vector_hash="sha256:" + f"{index:064x}")
        for index in range(count)
    ]


def build(
    kind: StateAuditCandidateKind = StateAuditCandidateKind.SELECTED_STRENGTH,
    trial_id: str = "sa_00000",
    count: int = 8,
    layer: int = 13,
    master_seed: int = 20260727,
    templates=None,
) -> StateAuditCandidateSet:
    refs = directions(count)
    if templates is None:
        if kind is StateAuditCandidateKind.SELECTED_STRENGTH:
            templates = selected_strength_templates(refs, 0.10, 0.10 * REFERENCE_NORM)
        else:
            templates = calibration_grid_templates(
                refs, [(ratio, ratio * REFERENCE_NORM) for ratio in RATIOS]
            )
    return build_state_audit_candidate_set(
        trial_id=trial_id,
        study_id="bluedot_state_dependence",
        variant_id=f"{trial_id}.neutral_a",
        group_id=f"g{trial_id}",
        prompt_role=PromptRole.SMOKE,
        kind=kind,
        layer=layer,
        position_index=-1,
        direction_family_id="fam_v1",
        direction_family_hash=FAMILY_HASH,
        direction_count=count,
        templates=templates,
        master_seed=master_seed,
    )


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_selected_strength_builds_seventeen_candidates() -> None:
    candidate_set = build()
    assert len(candidate_set.candidates) == 17
    assert candidate_set.kind is StateAuditCandidateKind.SELECTED_STRENGTH
    assert candidate_set.norm_ratios == [0.10]
    assert sum(1 for c in candidate_set.candidates if not c.is_noop) == 16


def test_calibration_grid_builds_eighty_one_candidates() -> None:
    candidate_set = build(kind=StateAuditCandidateKind.CALIBRATION_GRID)
    assert len(candidate_set.candidates) == 81
    assert candidate_set.norm_ratios == list(RATIOS)
    assert sum(1 for c in candidate_set.candidates if not c.is_noop) == 80


def test_the_calibration_grid_carries_one_shared_noop_not_one_per_ratio() -> None:
    templates = calibration_grid_templates(
        directions(), [(ratio, ratio * REFERENCE_NORM) for ratio in RATIOS]
    )
    assert sum(1 for template in templates if template.is_noop) == 1


@pytest.mark.parametrize(
    "kind", [StateAuditCandidateKind.SELECTED_STRENGTH, StateAuditCandidateKind.CALIBRATION_GRID]
)
def test_every_set_has_exactly_one_noop(kind: StateAuditCandidateKind) -> None:
    candidate_set = build(kind=kind)
    noops = [c for c in candidate_set.candidates if c.is_noop]
    assert len(noops) == 1
    assert noops[0].sign == 0
    assert noops[0].strength == 0.0
    assert noops[0].global_alpha == 0.0
    assert noops[0].norm_ratio == 0.0
    assert noops[0].direction_ref is None


@pytest.mark.parametrize(
    "kind", [StateAuditCandidateKind.SELECTED_STRENGTH, StateAuditCandidateKind.CALIBRATION_GRID]
)
def test_every_direction_appears_at_both_signs_at_every_ratio(
    kind: StateAuditCandidateKind,
) -> None:
    candidate_set = build(kind=kind)
    seen = {
        (c.direction_ref, c.norm_ratio, c.sign) for c in candidate_set.candidates if not c.is_noop
    }
    expected = {
        (direction.opaque_id, ratio, sign)
        for direction in directions()
        for ratio in candidate_set.norm_ratios
        for sign in (1, -1)
    }
    assert seen == expected


def test_strength_is_sign_times_the_global_alpha() -> None:
    candidate_set = build()
    for candidate in candidate_set.candidates:
        if candidate.is_noop:
            continue
        assert candidate.strength == pytest.approx(candidate.sign * candidate.global_alpha)
        assert candidate.global_alpha == pytest.approx(0.10 * REFERENCE_NORM)


def test_a_selected_strength_set_uses_one_alpha_for_every_candidate() -> None:
    candidate_set = build()
    alphas = {c.global_alpha for c in candidate_set.candidates if not c.is_noop}
    assert alphas == {0.10 * REFERENCE_NORM}


def test_the_calibration_grid_uses_one_alpha_per_ratio() -> None:
    candidate_set = build(kind=StateAuditCandidateKind.CALIBRATION_GRID)
    by_ratio: dict[float, set[float]] = {}
    for candidate in candidate_set.candidates:
        if candidate.is_noop:
            continue
        by_ratio.setdefault(candidate.norm_ratio, set()).add(candidate.global_alpha)
    assert by_ratio == {ratio: {ratio * REFERENCE_NORM} for ratio in RATIOS}


# ---------------------------------------------------------------------------
# Opacity and leakage
# ---------------------------------------------------------------------------


LEAKY_TERMS = (
    "answer_token",
    "random_orthogonal",
    "random_control",
    "direction_positive",
    "direction_negative",
    "noop_control",
    "construction_role",
    "analysis_role",
    "centered",
    "unembed",
)


def test_candidate_ids_are_opaque_and_positional_only() -> None:
    candidate_set = build()
    for index, candidate in enumerate(candidate_set.candidates):
        assert candidate.candidate_id == f"sa_00000.opaque_{index:02d}"


def test_the_serialized_candidate_set_names_no_construction_role() -> None:
    payload = canonical_json_bytes(build().model_dump(mode="json")).decode("utf-8").lower()
    for term in (*LEAKY_TERMS, "answer", "family_role", "semantic"):
        assert term not in payload, f"{term!r} leaked into the candidate set"


def test_the_public_view_hides_the_direction_the_ratio_and_the_mechanism() -> None:
    candidate_set = build()
    for candidate, view in zip(
        candidate_set.candidates, public_candidate_views(candidate_set), strict=True
    ):
        assert set(view) == {
            "candidate_id",
            "operation",
            "layer",
            "position_index",
            "strength",
        }
        assert view["operation"] == "residual_add"
        assert view["strength"] == candidate.strength


def test_a_noop_and_a_signed_candidate_have_the_same_public_shape() -> None:
    """A control and a steer differ only in their published strength, which is the design."""
    candidate_set = build()
    noop = next(c for c in candidate_set.candidates if c.is_noop)
    signed = next(c for c in candidate_set.candidates if not c.is_noop)
    assert set(public_view(noop)) == set(public_view(signed))
    assert public_view(noop)["operation"] == public_view(signed)["operation"]


def test_a_candidate_id_naming_a_construction_role_is_refused() -> None:
    with pytest.raises(ValidationError, match="opaque"):
        StateAuditCandidate(
            candidate_id="trial.answer_token_A",
            order_index=0,
            is_noop=True,
            sign=0,
            layer=13,
            position_index=-1,
            norm_ratio=0.0,
            global_alpha=0.0,
            strength=0.0,
            signed_intervention_hash="sha256:" + "0" * 64,
        )


def test_a_direction_ref_naming_a_family_is_refused() -> None:
    with pytest.raises(ValidationError, match="opaque"):
        StateAuditCandidate(
            candidate_id="trial.opaque_00",
            order_index=0,
            is_noop=False,
            direction_ref="random_control_02",
            direction_vector_hash="sha256:" + "1" * 64,
            sign=1,
            layer=13,
            position_index=-1,
            norm_ratio=0.10,
            global_alpha=3.2,
            strength=3.2,
            signed_intervention_hash="sha256:" + "0" * 64,
        )


def test_the_private_analysis_role_names_no_direction_family() -> None:
    candidate_set = build()
    roles = {candidate_analysis_role(c) for c in candidate_set.candidates}
    for role in roles:
        for term in ("answer", "random", "positive", "negative"):
            assert term not in role


# ---------------------------------------------------------------------------
# Ordering, determinism, and hashing
# ---------------------------------------------------------------------------


def test_ordering_is_deterministic_from_the_seed_and_the_trial() -> None:
    first, second = build(), build()
    assert [c.signed_intervention_hash for c in first.candidates] == [
        c.signed_intervention_hash for c in second.candidates
    ]
    assert first.candidate_set_hash == second.candidate_set_hash
    assert first.order_seed == candidate_order_seed(
        20260727, "sa_00000", StateAuditCandidateKind.SELECTED_STRENGTH
    )


def test_a_different_trial_gets_a_different_order() -> None:
    first = build(trial_id="sa_00000")
    second = build(trial_id="sa_00001")
    assert first.order_seed != second.order_seed
    first_layout = [c.signed_intervention_hash for c in first.candidates]
    second_layout = [c.signed_intervention_hash for c in second.candidates]
    assert first_layout != second_layout
    assert sorted(first_layout) == sorted(second_layout)


def test_position_does_not_encode_sign() -> None:
    """Ids are assigned after the shuffle, so the layout must not be sign-sorted."""
    signs = [c.sign for c in build().candidates if not c.is_noop]
    assert signs != sorted(signs, reverse=True)
    assert signs != sorted(signs)


def test_the_set_hash_covers_the_candidates() -> None:
    candidate_set = build()
    dumped = candidate_set.model_dump(mode="json")
    assert compute_state_audit_candidate_set_hash(dumped) == candidate_set.candidate_set_hash

    dumped["candidates"][0]["strength"] = 99.0
    assert compute_state_audit_candidate_set_hash(dumped) != candidate_set.candidate_set_hash


def test_the_set_hash_ignores_the_creation_timestamp() -> None:
    candidate_set = build()
    dumped = candidate_set.model_dump(mode="json")
    dumped["created_at"] = "2001-01-01T00:00:00Z"
    assert compute_state_audit_candidate_set_hash(dumped) == candidate_set.candidate_set_hash


def test_an_edited_candidate_set_does_not_load() -> None:
    dumped = json.loads(canonical_json_bytes(build().model_dump(mode="json")))
    dumped["candidates"][3]["strength"] = dumped["candidates"][3]["strength"] + 1.0
    with pytest.raises(ValidationError):
        StateAuditCandidateSet.model_validate(dumped)


def test_the_signed_intervention_hash_covers_the_vector_and_the_strength() -> None:
    base = signed_intervention_hash("sha256:" + "1" * 64, 1, 13, -1, 0.10, 3.2)
    assert base != signed_intervention_hash("sha256:" + "2" * 64, 1, 13, -1, 0.10, 3.2)
    assert base != signed_intervention_hash("sha256:" + "1" * 64, -1, 13, -1, 0.10, 3.2)
    assert base != signed_intervention_hash("sha256:" + "1" * 64, 1, 20, -1, 0.10, 3.2)
    assert base != signed_intervention_hash("sha256:" + "1" * 64, 1, 13, -1, 0.20, 3.2)
    assert base == signed_intervention_hash("sha256:" + "1" * 64, 1, 13, -1, 0.10, 3.2)


def test_two_candidates_applying_the_same_vector_share_an_intervention_hash() -> None:
    first = build(trial_id="sa_00000")
    second = build(trial_id="sa_00001")
    assert {c.signed_intervention_hash for c in first.candidates} == {
        c.signed_intervention_hash for c in second.candidates
    }


# ---------------------------------------------------------------------------
# Builder guards
# ---------------------------------------------------------------------------


def test_a_repeated_direction_id_is_refused() -> None:
    duplicated = [directions(1)[0], directions(1)[0]]
    with pytest.raises(CandidateBuildError, match="repeats an opaque id"):
        selected_strength_templates(duplicated, 0.10, 3.2)


def test_two_ids_for_the_same_vector_are_refused() -> None:
    shared = "sha256:" + "3" * 64
    refs = [
        DirectionRef(opaque_id="bd1.aaaa", vector_hash=shared),
        DirectionRef(opaque_id="bd1.bbbb", vector_hash=shared),
    ]
    with pytest.raises(CandidateBuildError, match="same vector hash"):
        selected_strength_templates(refs, 0.10, 3.2)


def test_a_nonpositive_alpha_is_refused() -> None:
    with pytest.raises(CandidateBuildError, match="positive ratio and alpha"):
        selected_strength_templates(directions(), 0.10, 0.0)


def test_a_descending_ratio_grid_is_refused() -> None:
    with pytest.raises(CandidateBuildError, match="ascending"):
        calibration_grid_templates(directions(), [(0.40, 12.8), (0.02, 0.64)])


def test_a_selected_strength_set_refuses_more_than_one_ratio() -> None:
    templates = calibration_grid_templates(directions(), [(0.05, 1.6), (0.10, 3.2)])
    with pytest.raises(ValidationError, match="exactly one ratio"):
        build(kind=StateAuditCandidateKind.SELECTED_STRENGTH, templates=templates)


def test_a_set_without_a_noop_is_refused() -> None:
    templates = [t for t in selected_strength_templates(directions(), 0.10, 3.2) if not t.is_noop]
    with pytest.raises(ValidationError, match="no-op candidates"):
        build(templates=templates)


def test_a_set_missing_one_sign_is_refused() -> None:
    templates = selected_strength_templates(directions(), 0.10, 3.2)
    reduced = [templates[0], *templates[2:]]
    with pytest.raises(ValidationError, match="signed candidates"):
        build(templates=reduced)


def test_a_per_prompt_alpha_inside_one_set_is_refused() -> None:
    """The leak the global rule exists to prevent, refused at the record boundary."""
    templates = list(selected_strength_templates(directions(), 0.10, 3.2))
    templates[0] = replace(templates[0], global_alpha=9.9)
    with pytest.raises(ValidationError, match="more than one alpha"):
        build(templates=templates)


def test_the_mechanism_mapping_is_explicit() -> None:
    candidate_set = build()
    noop = next(c for c in candidate_set.candidates if c.is_noop)
    signed = next(c for c in candidate_set.candidates if not c.is_noop)
    assert candidate_mechanism(noop) is Mechanism.NOOP
    assert candidate_mechanism(signed) is Mechanism.RESIDUAL_ADD


# ---------------------------------------------------------------------------
# The benchmark builder is untouched
# ---------------------------------------------------------------------------


def test_the_benchmark_four_candidate_builder_is_unchanged() -> None:
    """S4 adds a builder; it must not alter the one the original study depends on."""
    from causal_self_forecasting.trials.candidates import (
        build_candidate_set,
        default_templates,
    )
    from causal_self_forecasting.trials.candidates import public_view as benchmark_public_view

    templates = default_templates("d", "r", layer=2, position_index=-1, strength=1.5)
    assert len(templates) == 4
    assert [template.role for template in templates] == [
        "direction_positive",
        "direction_negative",
        "random_control",
        "noop_control",
    ]

    benchmark_set = build_candidate_set("trial_00000", 12345, templates)
    assert len(benchmark_set.candidates) == 4
    assert [c.intervention_id for c in benchmark_set.candidates] == [
        f"trial_00000.opaque_{index:02d}" for index in range(4)
    ]
    assert set(benchmark_public_view(benchmark_set.candidates[0])) == {
        "intervention_id",
        "operation",
        "layer",
        "position_index",
        "strength",
    }
