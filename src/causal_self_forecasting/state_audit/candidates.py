"""Study-specific candidate construction for the state-dependence arm.

Two shapes, built by two explicit functions rather than by one function with a flag, because
handing a calibration grid to a training run would silently change what the run measures:

* **selected strength**: 8 directions x 2 signs at one ratio, plus a no-op. 17 candidates. Used
  by smoke, training, and final test.
* **calibration grid**: 8 directions x 2 signs x 5 ratios, plus one shared no-op. 81 candidates.
  Used by calibration and by nothing else.

The benchmark's `trials/candidates.py` is untouched. Its four-candidate builder still produces
exactly what it produced before, and this module does not import from it beyond the public
operation naming, which both share so that a study candidate and a benchmark candidate look the
same to anything reading a public view.

Privacy is structural. A candidate holds an opaque direction id, that direction's content hash,
a sign, a layer, a ratio, and an alpha. It never holds a construction role, an answer label, a
random-control label, a semantic family, or the private analysis role, and the record type
refuses ids that name any of them. The public view narrows that further to the form of the
operation: a signed answer-token addition and a signed random-control addition are both a
residual addition at some layer and strength, which is all they visibly are.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..hashing import hash_object
from ..reproducibility import derive_seed
from ..schemas import (
    Mechanism,
    PromptRole,
    StateAuditCandidate,
    StateAuditCandidateKind,
    StateAuditCandidateSet,
    compute_state_audit_candidate_set_hash,
)

CANDIDATE_ALGORITHM_VERSION = "bluedot_state_audit_candidates_v1.0"

# Namespaced so that changing another seeded step cannot shift candidate ordering.
CANDIDATE_ORDER_LABEL = "bluedot.candidate_order"
SIGNED_INTERVENTION_LABEL = "bluedot.signed_intervention"

# What a no-op is called in an observation's `direction_ref`. Opaque enough: it names no
# construction role, and `is_noop` already says the same thing in its own field.
NOOP_DIRECTION_REF = "noop"

# Private analysis roles. Recorded on the `InterventionSpec` the harness executes, which is not
# a predictor-facing record, and deliberately worded so that neither names a direction family.
SIGNED_ANALYSIS_ROLE = "state_audit_signed_direction"
NOOP_ANALYSIS_ROLE = "state_audit_noop_integrity"

SIGNS: tuple[int, ...] = (1, -1)


class CandidateBuildError(ValueError):
    """Raised when a study candidate set cannot be built as specified."""


@dataclass(frozen=True)
class DirectionRef:
    """One direction as the candidate layer sees it: an opaque id and a content hash."""

    opaque_id: str
    vector_hash: str


@dataclass(frozen=True)
class CandidateTemplate:
    """A candidate before it is shuffled and given an opaque id."""

    is_noop: bool
    direction_ref: str | None
    direction_vector_hash: str | None
    sign: int
    norm_ratio: float
    global_alpha: float

    @property
    def strength(self) -> float:
        return float(self.sign * self.global_alpha)


def signed_intervention_hash(
    direction_vector_hash: str | None,
    sign: int,
    layer: int,
    position_index: int,
    norm_ratio: float,
    global_alpha: float,
) -> str:
    """Hash the actual signed intervention this candidate applies.

    Covers the direction's contents rather than its id, so two candidates that apply the same
    vector at the same strength hash the same however their ids were assigned, and a candidate
    whose stored vector changed underneath it stops matching.
    """
    return hash_object(
        {
            "label": SIGNED_INTERVENTION_LABEL,
            "algorithm_version": CANDIDATE_ALGORITHM_VERSION,
            "direction_vector_hash": direction_vector_hash,
            "sign": int(sign),
            "layer": int(layer),
            "position_index": int(position_index),
            "norm_ratio": float(norm_ratio),
            "global_alpha": float(global_alpha),
            "strength": float(sign * global_alpha),
        }
    )


def _validate_directions(directions: Sequence[DirectionRef]) -> list[DirectionRef]:
    if not directions:
        raise CandidateBuildError("a candidate set needs at least one direction")
    ids = [direction.opaque_id for direction in directions]
    if len(set(ids)) != len(ids):
        raise CandidateBuildError(f"the direction list repeats an opaque id: {sorted(ids)}")
    hashes = [direction.vector_hash for direction in directions]
    if len(set(hashes)) != len(hashes):
        raise CandidateBuildError(
            "two directions carry the same vector hash, so they are the same vector under two "
            "ids; that would make one sign pair a duplicate of another"
        )
    return list(directions)


def _noop_template() -> CandidateTemplate:
    return CandidateTemplate(
        is_noop=True,
        direction_ref=None,
        direction_vector_hash=None,
        sign=0,
        norm_ratio=0.0,
        global_alpha=0.0,
    )


def selected_strength_templates(
    directions: Sequence[DirectionRef],
    norm_ratio: float,
    global_alpha: float,
) -> list[CandidateTemplate]:
    """8 directions x 2 signs at one ratio, plus a no-op: 17 candidates.

    The shape smoke, training, and final test use. One ratio, one alpha, both signs for every
    direction, and the no-op retained as a live integrity control rather than dropped once
    calibration has finished with it.
    """
    if norm_ratio <= 0.0 or global_alpha <= 0.0:
        raise CandidateBuildError(
            f"a selected-strength set needs a positive ratio and alpha, got ratio {norm_ratio} "
            f"and alpha {global_alpha}"
        )
    resolved = _validate_directions(directions)
    templates = [
        CandidateTemplate(
            is_noop=False,
            direction_ref=direction.opaque_id,
            direction_vector_hash=direction.vector_hash,
            sign=sign,
            norm_ratio=float(norm_ratio),
            global_alpha=float(global_alpha),
        )
        for direction in resolved
        for sign in SIGNS
    ]
    templates.append(_noop_template())
    return templates


def calibration_grid_templates(
    directions: Sequence[DirectionRef],
    ratio_alphas: Sequence[tuple[float, float]],
) -> list[CandidateTemplate]:
    """8 directions x 2 signs x 5 ratios, plus one shared no-op: 81 candidates.

    One no-op, not one per ratio: adding zero does the same thing whatever ratio is under test,
    and five identical controls would inflate the no-op count without adding evidence.

    Used only by calibration. Handing this shape to training or final test would run those roles
    across the whole grid instead of at the one selected strength.
    """
    resolved = _validate_directions(directions)
    if not ratio_alphas:
        raise CandidateBuildError("the calibration grid needs at least one ratio")
    ratios = [float(ratio) for ratio, _ in ratio_alphas]
    if ratios != sorted(ratios):
        raise CandidateBuildError(f"the ratio grid must be ascending, got {ratios}")
    if len(set(ratios)) != len(ratios):
        raise CandidateBuildError(f"the ratio grid repeats a ratio: {ratios}")
    for ratio, alpha in ratio_alphas:
        if ratio <= 0.0 or alpha <= 0.0:
            raise CandidateBuildError(
                f"ratio {ratio} with alpha {alpha}: both must be positive on the grid"
            )

    templates = [
        CandidateTemplate(
            is_noop=False,
            direction_ref=direction.opaque_id,
            direction_vector_hash=direction.vector_hash,
            sign=sign,
            norm_ratio=float(ratio),
            global_alpha=float(alpha),
        )
        for ratio, alpha in ratio_alphas
        for direction in resolved
        for sign in SIGNS
    ]
    templates.append(_noop_template())
    return templates


def candidate_order_seed(master_seed: int, trial_id: str, kind: StateAuditCandidateKind) -> int:
    """The seed that fixes one trial's candidate order."""
    return derive_seed(
        CANDIDATE_ORDER_LABEL,
        master_seed,
        trial_id,
        kind.value,
        CANDIDATE_ALGORITHM_VERSION,
    )


def build_state_audit_candidate_set(
    trial_id: str,
    study_id: str,
    variant_id: str,
    group_id: str,
    prompt_role: PromptRole,
    kind: StateAuditCandidateKind,
    layer: int,
    position_index: int,
    direction_family_id: str,
    direction_family_hash: str,
    direction_count: int,
    templates: Sequence[CandidateTemplate],
    master_seed: int,
) -> StateAuditCandidateSet:
    """Shuffle templates into a hashed candidate set with opaque ids.

    Ids are assigned after the shuffle, so a candidate's position in the list says nothing about
    which direction or which sign it holds. The order is a pure function of the master seed and
    the trial id, so it is reproducible without being predictable from the candidate itself.
    """
    if len(templates) < 2:
        raise CandidateBuildError(f"trial {trial_id}: a candidate set needs at least two entries")

    order_seed = candidate_order_seed(master_seed, trial_id, kind)
    shuffled = list(templates)
    random.Random(order_seed).shuffle(shuffled)

    candidates: list[StateAuditCandidate] = []
    for index, template in enumerate(shuffled):
        candidates.append(
            StateAuditCandidate(
                candidate_id=f"{trial_id}.opaque_{index:02d}",
                order_index=index,
                is_noop=template.is_noop,
                direction_ref=template.direction_ref,
                direction_vector_hash=template.direction_vector_hash,
                sign=template.sign,
                layer=layer,
                position_index=position_index,
                norm_ratio=template.norm_ratio,
                global_alpha=template.global_alpha,
                strength=template.strength,
                signed_intervention_hash=signed_intervention_hash(
                    template.direction_vector_hash,
                    template.sign,
                    layer,
                    position_index,
                    template.norm_ratio,
                    template.global_alpha,
                ),
            )
        )

    ratios = sorted({c.norm_ratio for c in candidates if not c.is_noop})
    draft: dict[str, Any] = {
        "schema_version": StateAuditCandidateSet.model_fields["schema_version"].default,
        "trial_id": trial_id,
        "study_id": study_id,
        "variant_id": variant_id,
        "group_id": group_id,
        "prompt_role": prompt_role.value,
        "kind": kind.value,
        "layer": layer,
        "position_index": position_index,
        "direction_family_id": direction_family_id,
        "direction_family_hash": direction_family_hash,
        "direction_count": direction_count,
        "norm_ratios": ratios,
        "order_seed": order_seed,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }

    return StateAuditCandidateSet(
        trial_id=trial_id,
        study_id=study_id,
        variant_id=variant_id,
        group_id=group_id,
        prompt_role=prompt_role,
        kind=kind,
        layer=layer,
        position_index=position_index,
        direction_family_id=direction_family_id,
        direction_family_hash=direction_family_hash,
        direction_count=direction_count,
        norm_ratios=ratios,
        order_seed=order_seed,
        candidates=candidates,
        candidate_set_hash=compute_state_audit_candidate_set_hash(draft),
    )


def candidate_mechanism(candidate: StateAuditCandidate) -> Mechanism:
    """The harness mechanism one candidate executes.

    A signed direction is a residual addition and a no-op is a no-op. The public view maps both
    to `residual_add`, which is what makes a random control indistinguishable from an
    answer-token direction to anything reading the public description.
    """
    return Mechanism.NOOP if candidate.is_noop else Mechanism.RESIDUAL_ADD


def candidate_analysis_role(candidate: StateAuditCandidate) -> str:
    """The private role recorded on the executed spec. Never published."""
    return NOOP_ANALYSIS_ROLE if candidate.is_noop else SIGNED_ANALYSIS_ROLE


def public_view(candidate: StateAuditCandidate) -> dict[str, Any]:
    """What a predictor without payload access may see.

    The operation, the layer, the position, and the strength. Not the direction id, not the
    vector hash, not the sign as a separate field, not the ratio, and not the mechanism: the
    strength already carries the sign, and everything else would either name the direction or
    let a method tell a control from a steer without reading the model at all.
    """
    return {
        "candidate_id": candidate.candidate_id,
        "operation": "residual_add",
        "layer": candidate.layer,
        "position_index": candidate.position_index,
        "strength": candidate.strength,
    }


def public_candidate_views(candidate_set: StateAuditCandidateSet) -> list[dict[str, Any]]:
    return [public_view(candidate) for candidate in candidate_set.candidates]


__all__ = [
    "CANDIDATE_ALGORITHM_VERSION",
    "CANDIDATE_ORDER_LABEL",
    "NOOP_ANALYSIS_ROLE",
    "NOOP_DIRECTION_REF",
    "SIGNED_ANALYSIS_ROLE",
    "SIGNS",
    "CandidateBuildError",
    "CandidateTemplate",
    "DirectionRef",
    "build_state_audit_candidate_set",
    "calibration_grid_templates",
    "candidate_analysis_role",
    "candidate_mechanism",
    "candidate_order_seed",
    "public_candidate_views",
    "public_view",
    "selected_strength_templates",
    "signed_intervention_hash",
]
